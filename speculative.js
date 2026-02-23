/**
 * PureBee — 6 — Self-Speculative Decoding
 *
 * Generates tokens faster by drafting with early exit (fewer layers)
 * and verifying with the full model. Same model, different depths.
 *
 * How it works:
 *   1. DRAFT: Run forward pass with only K layers (e.g., 4 of 16)
 *      Generate N draft tokens quickly (fewer disk reads, less compute)
 *   2. VERIFY: Run full forward pass on all N draft tokens
 *      Compare draft predictions vs full-model predictions
 *   3. ACCEPT/REJECT: Accept matching tokens, reject from first mismatch
 *      Always get at least 1 correct token per round
 *
 * Why self-speculative (same model)?
 *   - No tokenizer mismatch (different models use different tokenizers)
 *   - No separate model to load/manage
 *   - Early layers often predict common tokens correctly
 *   - Verification amortizes disk I/O: load all 16 layers once for N tokens
 *
 * Expected speedup: 1.5-3x when acceptance rate is high.
 *
 * Zero external dependencies.
 */

'use strict';

class SelfSpeculativeDecoder {
  /**
   * @param {StreamingLlamaRuntime} runtime — the streaming LLaMA runtime
   * @param {Object} options
   * @param {number} options.draftLayers — layers to use for drafting (default 4)
   * @param {number} options.draftCount — draft tokens per round (default 4)
   * @param {number} options.temperature — sampling temperature
   * @param {number} options.topK — top-K sampling
   */
  constructor(runtime, options = {}) {
    this.runtime = runtime;
    this.draftLayers = options.draftLayers || 4;
    this.draftCount = options.draftCount || 4;
    this.temperature = options.temperature || 0.6;
    this.topK = options.topK || 40;

    // Stats
    this._totalRounds = 0;
    this._totalDrafted = 0;
    this._totalAccepted = 0;
  }

  /**
   * Generate tokens using self-speculative decoding.
   *
   * @param {number[]} promptTokens
   * @param {number} maxTokens
   * @param {Object} opts
   * @returns {{ tokens, generated, tokPerSec, acceptanceRate, prefillTime, decodeTime }}
   */
  generate(promptTokens, maxTokens = 50, opts = {}) {
    const temperature = opts.temperature || this.temperature;
    const topK = opts.topK || this.topK;
    const eosIds = opts.eosIds || [2];
    const onToken = opts.onToken || null;
    const greedy = opts.greedy || false;

    const runtime = this.runtime;
    runtime.resetCache();

    const allTokens = [...promptTokens];
    let pos = 0;

    // ── PREFILL ──
    const prefillStart = Date.now();
    const { logits: prefillLogits } = runtime.forward(promptTokens, 0);
    const prefillTime = Date.now() - prefillStart;
    pos = promptTokens.length;

    // First generated token from prefill
    let nextToken = greedy
      ? runtime.argmax(prefillLogits)
      : runtime.sample(prefillLogits, topK, temperature);
    allTokens.push(nextToken);
    if (onToken) onToken(nextToken, 0);

    let generated = 1;
    let hitEos = eosIds.includes(nextToken);

    // ── SPECULATIVE DECODE LOOP ──
    const decodeStart = Date.now();

    while (generated < maxTokens && !hitEos) {
      // ── DRAFT PHASE: Generate K tokens using only first N layers ──
      const draftTokens = [nextToken];
      const draftLogits = [];
      let draftPos = pos;

      for (let d = 0; d < this.draftCount && generated + d < maxTokens; d++) {
        const currentToken = draftTokens[draftTokens.length - 1];
        if (eosIds.includes(currentToken)) break;

        const { logits } = runtime.forward([currentToken], draftPos, {
          maxLayers: this.draftLayers,
        });
        draftPos++;

        const drafted = greedy
          ? runtime.argmax(logits)
          : runtime.sample(logits, topK, temperature);

        draftLogits.push(logits);
        draftTokens.push(drafted);
      }

      const numDrafted = draftTokens.length - 1; // exclude the input token
      if (numDrafted === 0) {
        // No drafts possible (hit EOS on input), just do standard forward
        break;
      }

      this._totalDrafted += numDrafted;
      this._totalRounds++;

      // ── VERIFY PHASE: Run full model on draft tokens ──
      // Rollback KV cache to position before drafts
      runtime.rollbackCache(pos);

      // Verify each draft token one at a time with full model
      let accepted = 0;

      for (let d = 0; d < numDrafted; d++) {
        const verifyToken = draftTokens[d]; // input token for this position
        const { logits: verifyLogits } = runtime.forward([verifyToken], pos + d);

        const verifyPred = greedy
          ? runtime.argmax(verifyLogits)
          : runtime.sample(verifyLogits, topK, temperature);

        const draftPred = draftTokens[d + 1]; // what the draft predicted

        if (verifyPred === draftPred) {
          // Draft was correct — accept
          accepted++;
          allTokens.push(draftPred);
          generated++;
          if (onToken) onToken(draftPred, generated - 1);

          if (eosIds.includes(draftPred)) {
            hitEos = true;
            break;
          }
        } else {
          // Draft was wrong — use verified token instead, reject rest
          allTokens.push(verifyPred);
          generated++;
          if (onToken) onToken(verifyPred, generated - 1);

          if (eosIds.includes(verifyPred)) {
            hitEos = true;
          }

          // Rollback cache past rejected positions
          runtime.rollbackCache(pos + d + 2);
          break;
        }
      }

      // If all drafts were accepted, we need one more verified token
      if (accepted === numDrafted && !hitEos) {
        const lastDraftToken = draftTokens[draftTokens.length - 1];
        const { logits: bonusLogits } = runtime.forward([lastDraftToken], pos + numDrafted);
        const bonusToken = greedy
          ? runtime.argmax(bonusLogits)
          : runtime.sample(bonusLogits, topK, temperature);

        allTokens.push(bonusToken);
        generated++;
        if (onToken) onToken(bonusToken, generated - 1);
        if (eosIds.includes(bonusToken)) hitEos = true;

        pos += numDrafted + 1;
        nextToken = bonusToken;
      } else {
        // Some were rejected
        pos += accepted + 1;
        nextToken = allTokens[allTokens.length - 1];
      }

      this._totalAccepted += accepted;
    }

    const decodeTime = Date.now() - decodeStart;
    const decodedTokens = Math.max(generated - 1, 1);
    const tokPerSec = decodedTokens / (decodeTime / 1000);

    return {
      tokens: allTokens,
      generated,
      prefillTime,
      decodeTime,
      tokPerSec,
      acceptanceRate: this._totalDrafted > 0
        ? (this._totalAccepted / this._totalDrafted * 100).toFixed(0) + '%'
        : 'N/A',
    };
  }

  /**
   * Get speculative decoding stats.
   */
  get stats() {
    return {
      totalRounds: this._totalRounds,
      totalDrafted: this._totalDrafted,
      totalAccepted: this._totalAccepted,
      acceptanceRate: this._totalDrafted > 0
        ? (this._totalAccepted / this._totalDrafted * 100).toFixed(0) + '%'
        : 'N/A',
    };
  }

  resetStats() {
    this._totalRounds = 0;
    this._totalDrafted = 0;
    this._totalAccepted = 0;
  }
}

module.exports = { SelfSpeculativeDecoder };
