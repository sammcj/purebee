/**
 * PureBee — 2 — BPE Tokenizer
 *
 * Loads and uses the SentencePiece-style tokenizer for LLaMA models.
 *
 * Tokenizer binary format (this specific tokenizer.bin):
 *   For each token (vocab_size times):
 *     len: int32 (little-endian)
 *     bytes: [len bytes] — the token string (UTF-8)
 *
 * Token layout:
 *   0: <unk>    — unknown token
 *   1: <s>      — begin of sequence (BOS)
 *   2: </s>     — end of sequence (EOS)
 *   3-258: byte fallbacks (byte 0x00 through 0xFF)
 *   259+: BPE merged tokens (most common first)
 *
 * Encoding uses BPE merge loop: start with byte-level tokens,
 * repeatedly merge the pair that forms the lowest-ID merged token
 * (lowest ID among merged tokens = most common = highest priority).
 *
 * Zero external dependencies.
 */

'use strict';

const fs = require('fs');

class BPETokenizer {
  constructor() {
    this.vocab = [];        // id → string
    this.tokenToId = {};    // string → id
    this.vocabSize = 0;
    this.maxTokenLen = 0;
    this._loaded = false;
  }

  /**
   * Load tokenizer from a tokenizer.bin file.
   *
   * @param {string} path - path to tokenizer.bin
   * @param {number} vocabSize - expected vocabulary size (from model config)
   */
  load(path, vocabSize) {
    console.log(`  [Tokenizer] Loading ${path}...`);
    const buffer = fs.readFileSync(path);

    let offset = 0;
    this.vocabSize = vocabSize;
    this.vocab = new Array(vocabSize);
    let maxLen = 0;

    // Read each token: int32 length + UTF-8 bytes
    for (let i = 0; i < vocabSize; i++) {
      if (offset + 4 > buffer.byteLength) {
        throw new Error(`Tokenizer file too short at token ${i}, offset ${offset}`);
      }

      const len = buffer.readInt32LE(offset);
      offset += 4;

      if (len < 0 || len > 1000) {
        throw new Error(`Invalid token length ${len} at token ${i}, offset ${offset - 4}`);
      }

      // Read raw bytes and convert to string
      const bytes = buffer.slice(offset, offset + len);
      const str = bytes.toString('utf-8');
      offset += len;

      this.vocab[i] = str;
      // Only map to ID if this is the first occurrence
      if (this.tokenToId[str] === undefined) {
        this.tokenToId[str] = i;
      }
      if (len > maxLen) maxLen = len;
    }

    this.maxTokenLen = maxLen;
    this._loaded = true;

    console.log(`  [Tokenizer] Loaded ${vocabSize} tokens, max_len=${maxLen}`);
  }

  /**
   * Encode a string into token IDs using BPE.
   *
   * Process:
   *   1. Convert text to UTF-8 bytes
   *   2. Map each byte to its byte-level token (token_id = byte_value + 3)
   *   3. Repeatedly merge the pair that produces the lowest token ID
   *      (lowest ID = most common merge = highest priority)
   *
   * @param {string} text
   * @returns {number[]}
   */
  encode(text) {
    if (!this._loaded) throw new Error('Tokenizer not loaded');
    if (text.length === 0) return [];

    // Step 1: Convert to UTF-8 bytes and map to byte-level tokens
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);

    let tokens = [];
    for (let i = 0; i < bytes.length; i++) {
      // Byte fallback tokens start at ID 3 (byte 0x00 = token 3, etc.)
      tokens.push(bytes[i] + 3);
    }

    // Step 2: BPE merge loop
    // Repeatedly find the pair of adjacent tokens whose merge exists
    // in the vocab with the LOWEST token ID (= most common merge)
    while (tokens.length >= 2) {
      let bestId = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < tokens.length - 1; i++) {
        const merged = this.vocab[tokens[i]] + this.vocab[tokens[i + 1]];
        const mergedId = this.tokenToId[merged];
        if (mergedId !== undefined && mergedId < bestId) {
          bestId = mergedId;
          bestIdx = i;
        }
      }

      // No more merges possible
      if (bestIdx === -1) break;

      // Apply the merge
      tokens[bestIdx] = bestId;
      tokens.splice(bestIdx + 1, 1);
    }

    return tokens;
  }

  /**
   * Decode token IDs back to a string.
   *
   * @param {number[]} ids
   * @returns {string}
   */
  decode(ids) {
    if (!this._loaded) throw new Error('Tokenizer not loaded');
    let pieces = [];
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] === this.bosId || ids[i] === this.eosId) continue;
      const token = this.vocab[ids[i]];
      if (token === undefined) {
        pieces.push('?');
      } else {
        pieces.push(token);
      }
    }
    return pieces.join('');
  }

  /**
   * Decode a single token ID to its string.
   * Handles BOS/EOS and leading space stripping.
   *
   * @param {number} id
   * @param {number} prevId
   * @returns {string}
   */
  decodeToken(id, prevId = -1) {
    if (id === this.bosId || id === this.eosId) return '';

    const token = this.vocab[id];
    if (token === undefined) return '';

    // Strip leading space after BOS
    if (prevId === this.bosId && token.startsWith(' ')) {
      return token.slice(1);
    }

    return token;
  }

  /** BOS (beginning of sequence) token ID */
  get bosId() { return 1; }

  /** EOS (end of sequence) token ID */
  get eosId() { return 2; }
}

module.exports = { BPETokenizer };
