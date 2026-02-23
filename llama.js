/**
 * PureBee — 2 — LLaMA Transformer Runtime
 *
 * A complete LLaMA-architecture transformer running on PureBee instructions.
 * Supports Karpathy's TinyStories models (stories15M/42M/110M).
 *
 * Architecture differences from GPT-2 (Phase 1):
 *   - RMSNorm instead of LayerNorm (simpler — no bias, no mean)
 *   - RoPE instead of learned position embeddings (rotary — encodes position in Q/K)
 *   - SiLU + SwiGLU instead of GELU (gate * up projection)
 *   - No bias in linear layers
 *   - KV-cache for efficient autoregressive generation
 *
 * Zero external dependencies. Pure PureBee instructions.
 */

'use strict';

const { PureBee } = require('./purebee');
const { Tensor } = require('./memory');

class LlamaConfig {
  constructor(opts) {
    this.dim       = opts.dim;         // model dimension
    this.hiddenDim = opts.hiddenDim;   // FFN intermediate dimension
    this.nLayers   = opts.nLayers;     // number of transformer blocks
    this.nHeads    = opts.nHeads;      // number of query attention heads
    this.nKvHeads  = opts.nKvHeads;    // number of key/value heads (GQA)
    this.vocabSize = opts.vocabSize;   // vocabulary size
    this.seqLen    = opts.seqLen;      // max sequence length
    this.headDim   = opts.headDim || Math.floor(this.dim / this.nHeads);
    this.ropeTheta = opts.ropeTheta || 10000.0; // RoPE base frequency
  }
}

class LlamaRuntime {
  constructor(config, options = {}) {
    this.config = config;
    this.gpu = new PureBee({ log: options.log || false });
    this._loaded = false;
    this._kvInitialized = false;

    // Initialize WASM SIMD acceleration
    if (options.noWasm) {
      this._wasmReady = false;
    } else {
      this._wasmReady = this.gpu.engine.initWasm();
    }
  }

  /**
   * Load real weights into PureBee memory.
   * Weights come from the model loader — already transposed for our matmul convention.
   *
   * @param {Object} weights - { name: Float32Array }
   * @param {boolean} sharedWeights - whether lm_head shares token_embedding
   */
  loadWeights(weights, sharedWeights = true) {
    const gpu = this.gpu;
    const { dim, hiddenDim, nLayers, nKvHeads, headDim, vocabSize, seqLen } = this.config;
    const kvDim = nKvHeads * headDim;

    console.log('  [LLaMA] Loading weights into PureBee memory...');

    // Helper: store either Float32Array or QuantizedTensor
    const store = (name, shape, data) => {
      const isQT = data && data.constructor && data.constructor.name === 'QuantizedTensor';
      if (isQT) {
        gpu.GRID_WRITE_RAW(name, data);
      } else {
        gpu.GRID_WRITE(name, shape, data);
      }
    };

    // Token embedding [vocabSize, dim] — always float32 (needed for lookup)
    gpu.GRID_WRITE('token_embedding', [vocabSize, dim], weights['token_embedding']);

    // Per-layer weights
    for (let l = 0; l < nLayers; l++) {
      // RMS norms — [dim] — always float32
      gpu.GRID_WRITE(`layer${l}.rms_att`, [dim], weights[`layer${l}.rms_att`]);
      gpu.GRID_WRITE(`layer${l}.rms_ffn`, [dim], weights[`layer${l}.rms_ffn`]);

      // Attention projections — may be quantized
      store(`layer${l}.wq`, [dim, dim], weights[`layer${l}.wq`]);
      store(`layer${l}.wk`, [dim, kvDim], weights[`layer${l}.wk`]);
      store(`layer${l}.wv`, [dim, kvDim], weights[`layer${l}.wv`]);
      store(`layer${l}.wo`, [dim, dim], weights[`layer${l}.wo`]);

      // FFN projections — may be quantized
      store(`layer${l}.w1`, [dim, hiddenDim], weights[`layer${l}.w1`]);
      store(`layer${l}.w2`, [hiddenDim, dim], weights[`layer${l}.w2`]);
      store(`layer${l}.w3`, [dim, hiddenDim], weights[`layer${l}.w3`]);
    }

    // Final RMS norm — [dim]
    gpu.GRID_WRITE('rms_final', [dim], weights['rms_final']);

    // LM head (if not shared)
    if (!sharedWeights && weights['lm_head']) {
      store('lm_head', [vocabSize, dim], weights['lm_head']);
    }

    this._sharedWeights = sharedWeights;
    this._loaded = true;

    // Cache float32 weight matrices in WASM memory for SIMD acceleration
    if (this._wasmReady) {
      const engine = this.gpu.engine;
      let cached = 0;
      for (let l = 0; l < nLayers; l++) {
        const weightNames = [
          `layer${l}.wq`, `layer${l}.wk`, `layer${l}.wv`, `layer${l}.wo`,
          `layer${l}.w1`, `layer${l}.w2`, `layer${l}.w3`,
        ];
        for (const name of weightNames) {
          const w = this.gpu._readRaw(name);
          // Only cache float32 Tensor weights (not QuantizedTensor)
          if (w instanceof Tensor && w.data instanceof Float32Array) {
            engine.cacheWeight(w);
            cached++;
          }
        }
      }
      console.log(`  [LLaMA] WASM SIMD: cached ${cached} weight matrices, ${engine.wasmStats.weightsMB}MB`);
    }

    const stats = this.gpu.stats();
    console.log(`  [LLaMA] ${stats.memory.tensors} tensors, ${stats.memory.totalMB}MB in PureBee memory`);
  }

  /**
   * Initialize KV cache for all layers.
   * Called once before generation begins.
   */
  _initKVCache() {
    const { nLayers, nKvHeads, headDim, seqLen } = this.config;
    const kvDim = nKvHeads * headDim;

    for (let l = 0; l < nLayers; l++) {
      this.gpu.GRID_ALLOC(`kv_k_${l}`, [seqLen, kvDim]).fill(0);
      this.gpu.GRID_ALLOC(`kv_v_${l}`, [seqLen, kvDim]).fill(0);
    }

    this._kvInitialized = true;
  }

  /**
   * Apply Rotary Position Embeddings (RoPE) to Q and K tensors.
   *
   * For each pair of dimensions (2i, 2i+1) at position pos:
   *   theta = 1 / (10000 ^ (2i / headDim))
   *   angle = pos * theta
   *   q_rot[2i]   = q[2i] * cos(angle) - q[2i+1] * sin(angle)
   *   q_rot[2i+1] = q[2i] * sin(angle) + q[2i+1] * cos(angle)
   *
   * RoPE encodes position WITHOUT learned embeddings. Elegant.
   */
  _applyRoPE(qData, kData, seqLen, startPos) {
    const { dim, nHeads, nKvHeads, headDim, ropeTheta } = this.config;
    const kvDim = nKvHeads * headDim;

    for (let t = 0; t < seqLen; t++) {
      const pos = startPos + t;

      for (let i = 0; i < headDim; i += 2) {
        const freq = 1.0 / Math.pow(ropeTheta, i / headDim);
        const angle = pos * freq;
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);

        // Rotate all Q heads
        for (let h = 0; h < nHeads; h++) {
          const idx = t * dim + h * headDim + i;
          const q0 = qData[idx];
          const q1 = qData[idx + 1];
          qData[idx]     = q0 * cosA - q1 * sinA;
          qData[idx + 1] = q0 * sinA + q1 * cosA;
        }

        // Rotate all K heads
        for (let h = 0; h < nKvHeads; h++) {
          const idx = t * kvDim + h * headDim + i;
          const k0 = kData[idx];
          const k1 = kData[idx + 1];
          kData[idx]     = k0 * cosA - k1 * sinA;
          kData[idx + 1] = k0 * sinA + k1 * cosA;
        }
      }
    }
  }

  /**
   * Forward pass through the LLaMA transformer.
   *
   * Supports two modes:
   *   - Prefill: process multiple tokens at once (startPos = 0, seqLen > 1)
   *   - Decode: process single new token using KV cache (startPos > 0, seqLen = 1)
   *
   * @param {number[]} tokenIds - input token IDs
   * @param {number} startPos - position in the sequence (for KV cache)
   * @returns {Float32Array} logits [vocabSize] for the last token
   */
  forward(tokenIds, startPos = 0) {
    if (!this._loaded) throw new Error('Weights not loaded');
    if (!this._kvInitialized) this._initKVCache();

    const gpu = this.gpu;
    const { dim, hiddenDim, nLayers, nHeads, nKvHeads, headDim, vocabSize } = this.config;
    const seqLen = tokenIds.length;
    const kvDim = nKvHeads * headDim;
    const cacheLen = startPos + seqLen; // total positions in cache

    // ── EMBEDDING LOOKUP ──
    const wte = gpu.GRID_READ('token_embedding');
    const xData = new Float32Array(seqLen * dim);
    for (let i = 0; i < seqLen; i++) {
      const tok = tokenIds[i];
      const srcOffset = tok * dim;
      const dstOffset = i * dim;
      for (let d = 0; d < dim; d++) {
        xData[dstOffset + d] = wte.data[srcOffset + d];
      }
    }
    gpu.GRID_WRITE('x', [seqLen, dim], xData);

    // ── TRANSFORMER BLOCKS ──
    for (let l = 0; l < nLayers; l++) {

      // ── Pre-attention RMS Norm ──
      gpu.RMS_NORM('x', `layer${l}.rms_att`, 'xnorm');

      // ── QKV Projections ──
      // q = xnorm @ wq → [seqLen, dim]
      // k = xnorm @ wk → [seqLen, kvDim]
      // v = xnorm @ wv → [seqLen, kvDim]
      gpu.LINEAR('xnorm', `layer${l}.wq`, null, 'q');
      gpu.LINEAR('xnorm', `layer${l}.wk`, null, 'k');
      gpu.LINEAR('xnorm', `layer${l}.wv`, null, 'v');

      // ── RoPE ──
      const qTensor = gpu.GRID_READ('q');
      const kTensor = gpu.GRID_READ('k');
      this._applyRoPE(qTensor.data, kTensor.data, seqLen, startPos);

      // ── Update KV Cache ──
      const vTensor = gpu.GRID_READ('v');
      const keyCache = gpu.GRID_READ(`kv_k_${l}`);
      const valCache = gpu.GRID_READ(`kv_v_${l}`);

      for (let t = 0; t < seqLen; t++) {
        const cachePos = startPos + t;
        const srcOffset = t * kvDim;
        const dstOffset = cachePos * kvDim;
        for (let d = 0; d < kvDim; d++) {
          keyCache.data[dstOffset + d] = kTensor.data[srcOffset + d];
          valCache.data[dstOffset + d] = vTensor.data[srcOffset + d];
        }
      }

      // ── Multi-Head Attention ──
      // For each query head, find the corresponding KV head (GQA support)
      const headsPerKvHead = nHeads / nKvHeads;
      const attnOut = new Float32Array(seqLen * dim);

      for (let t = 0; t < seqLen; t++) {
        for (let h = 0; h < nHeads; h++) {
          const kvH = Math.floor(h / headsPerKvHead);

          // Compute attention scores for this head at this position
          const scores = new Float32Array(cacheLen);
          const scale = 1.0 / Math.sqrt(headDim);

          for (let j = 0; j < cacheLen; j++) {
            // Causal mask: can only attend to positions <= current
            if (j > startPos + t) {
              scores[j] = -Infinity;
              continue;
            }
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += qTensor.data[t * dim + h * headDim + d] *
                     keyCache.data[j * kvDim + kvH * headDim + d];
            }
            scores[j] = dot * scale;
          }

          // Softmax over scores
          let maxScore = -Infinity;
          for (let j = 0; j < cacheLen; j++) {
            if (scores[j] > maxScore) maxScore = scores[j];
          }
          let sumExp = 0;
          for (let j = 0; j < cacheLen; j++) {
            scores[j] = Math.exp(scores[j] - maxScore);
            sumExp += scores[j];
          }
          for (let j = 0; j < cacheLen; j++) {
            scores[j] /= sumExp;
          }

          // Weighted sum of cached values
          for (let d = 0; d < headDim; d++) {
            let val = 0;
            for (let j = 0; j < cacheLen; j++) {
              val += scores[j] * valCache.data[j * kvDim + kvH * headDim + d];
            }
            attnOut[t * dim + h * headDim + d] = val;
          }
        }
      }

      gpu.GRID_WRITE('attn_concat', [seqLen, dim], attnOut);

      // ── Output Projection ──
      gpu.LINEAR('attn_concat', `layer${l}.wo`, null, 'attn_proj');

      // ── Residual Connection ──
      gpu.TENSOR_ADD('x', 'attn_proj', 'x');

      // ── Pre-FFN RMS Norm ──
      gpu.RMS_NORM('x', `layer${l}.rms_ffn`, 'xnorm');

      // ── SwiGLU Feed-Forward Network ──
      // gate = silu(xnorm @ w1)
      // up   = xnorm @ w3
      // hidden = gate * up
      // out  = hidden @ w2
      gpu.LINEAR('xnorm', `layer${l}.w1`, null, 'ff_gate');
      gpu.SILU('ff_gate', 'ff_gate_act');
      gpu.LINEAR('xnorm', `layer${l}.w3`, null, 'ff_up');
      gpu.ELEMENT_MUL('ff_gate_act', 'ff_up', 'ff_hidden');
      gpu.LINEAR('ff_hidden', `layer${l}.w2`, null, 'ff_out');

      // ── Residual Connection ──
      gpu.TENSOR_ADD('x', 'ff_out', 'x');
    }

    // ── Final RMS Norm ──
    gpu.RMS_NORM('x', 'rms_final', 'x_final');

    // ── LM Head — project to vocabulary ──
    // Use last token position only
    const xFinal = gpu.GRID_READ('x_final');
    const lastOffset = (seqLen - 1) * dim;

    // Extract last position's hidden state
    const lastHidden = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      lastHidden[d] = xFinal.data[lastOffset + d];
    }

    const lmWeight = this._sharedWeights
      ? gpu.GRID_READ('token_embedding')
      : gpu.GRID_READ('lm_head');

    // LM head: dot(hidden, W[v,:]) for each vocab entry
    // W is [vocabSize, dim] — a row-wise dot product (transposed matmul)
    const logits = new Float32Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      let dot = 0;
      const wOffset = v * dim;
      for (let d = 0; d < dim; d++) {
        dot += lastHidden[d] * lmWeight.data[wOffset + d];
      }
      logits[v] = dot;
    }

    gpu.SYNC();
    return logits;
  }

  /**
   * Reset KV cache. Call between independent generation sessions.
   */
  resetCache() {
    if (this._kvInitialized) {
      const { nLayers, nKvHeads, headDim, seqLen } = this.config;
      const kvDim = nKvHeads * headDim;
      for (let l = 0; l < nLayers; l++) {
        this.gpu.GRID_READ(`kv_k_${l}`).fill(0);
        this.gpu.GRID_READ(`kv_v_${l}`).fill(0);
      }
    }
  }

  /**
   * Sample next token from logits using top-k + temperature sampling.
   *
   * @param {Float32Array} logits
   * @param {number} topK
   * @param {number} temperature
   * @returns {number} token id
   */
  sample(logits, topK = 40, temperature = 0.8) {
    // Apply temperature
    const scaled = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;

    // Get top-K indices
    const indexed = Array.from(scaled).map((v, i) => [v, i]);
    indexed.sort((a, b) => b[0] - a[0]);
    const topKItems = indexed.slice(0, topK);

    // Softmax over top-K
    const maxV = topKItems[0][0];
    let sum = 0;
    const probs = topKItems.map(([v]) => {
      const e = Math.exp(v - maxV);
      sum += e;
      return e;
    });
    const normalized = probs.map(p => p / sum);

    // Sample from distribution
    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < normalized.length; i++) {
      cumulative += normalized[i];
      if (r <= cumulative) return topKItems[i][1];
    }
    return topKItems[0][1];
  }

  /**
   * Greedy argmax — pick the highest probability token.
   * More deterministic than sampling. Good for testing.
   *
   * @param {Float32Array} logits
   * @returns {number} token id
   */
  argmax(logits) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  /**
   * Generate text autoregressively.
   *
   * @param {number[]} promptTokens - encoded prompt
   * @param {number} maxTokens - max tokens to generate
   * @param {Object} opts - { temperature, topK, onToken }
   * @returns {{ tokens: number[], prefillTime: number, decodeTime: number, tokPerSec: number }}
   */
  generate(promptTokens, maxTokens = 50, opts = {}) {
    const temperature = opts.temperature || 0.8;
    const topK = opts.topK || 40;
    const onToken = opts.onToken || null;
    const eosId = opts.eosId || 2;
    const greedy = opts.greedy || false;

    this.resetCache();

    const allTokens = [...promptTokens];
    let pos = 0;

    // ── PREFILL: process all prompt tokens at once ──
    const prefillStart = Date.now();
    const logits = this.forward(promptTokens, 0);
    const prefillTime = Date.now() - prefillStart;
    pos = promptTokens.length;

    // Sample first generated token
    let nextToken = greedy ? this.argmax(logits) : this.sample(logits, topK, temperature);
    allTokens.push(nextToken);
    if (onToken) onToken(nextToken, 0);

    // ── DECODE: generate one token at a time using KV cache ──
    const decodeStart = Date.now();
    let generated = 1;

    for (let i = 1; i < maxTokens; i++) {
      if (nextToken === eosId) break;

      // Forward pass for single new token — KV cache handles history
      const stepLogits = this.forward([nextToken], pos);
      pos++;

      nextToken = greedy ? this.argmax(stepLogits) : this.sample(stepLogits, topK, temperature);
      allTokens.push(nextToken);
      generated++;

      if (onToken) onToken(nextToken, i);
      if (nextToken === eosId) break;
    }

    const decodeTime = Date.now() - decodeStart;
    const decodedTokens = Math.max(generated - 1, 1); // exclude first token (from prefill)
    const tokPerSec = decodedTokens / (decodeTime / 1000);

    return {
      tokens: allTokens,
      generated,
      prefillTime,
      decodeTime,
      tokPerSec,
    };
  }

  /** Cleanup. */
  shutdown() {
    // No-op in single-threaded mode. Reserved for future use.
  }
}

module.exports = { LlamaRuntime, LlamaConfig };
