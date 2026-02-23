/**
 * PureBee — GGUF Model Loader
 *
 * Parses GGUF v2/v3 model files and loads them into PureBee-compatible format.
 * Supports F32, F16, Q8_0, and Q4_0 tensor types.
 *
 * GGUF is the standard format used by llama.cpp. This loader opens up
 * thousands of open-source models for the PureBee runtime.
 *
 * Zero external dependencies.
 */

'use strict';

const fs = require('fs');

// ═══════════════════════════════════════════════════════════════════════════
// GGUF Constants
// ═══════════════════════════════════════════════════════════════════════════

const GGUF_MAGIC = 0x46554747; // "GGUF" as little-endian uint32

// Metadata value types
const GGUF_TYPE = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};

// Tensor data types
const GGML_TYPE = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
};

// Bytes per block for quantized types
const GGML_TYPE_SIZE = {
  [GGML_TYPE.F32]:  { blockSize: 1,   bytesPerBlock: 4 },
  [GGML_TYPE.F16]:  { blockSize: 1,   bytesPerBlock: 2 },
  [GGML_TYPE.Q4_0]: { blockSize: 32,  bytesPerBlock: 18 },  // f16 scale + 16 bytes (32 x 4-bit)
  [GGML_TYPE.Q4_1]: { blockSize: 32,  bytesPerBlock: 20 },  // f16 min + f16 scale + 16 bytes
  [GGML_TYPE.Q8_0]: { blockSize: 32,  bytesPerBlock: 34 },  // f16 scale + 32 bytes (32 x int8)
  [GGML_TYPE.Q8_1]: { blockSize: 32,  bytesPerBlock: 36 },  // f32 scale + f32 min + 32 bytes
  [GGML_TYPE.Q2_K]: { blockSize: 256, bytesPerBlock: 84 },
  [GGML_TYPE.Q3_K]: { blockSize: 256, bytesPerBlock: 110 },
  [GGML_TYPE.Q4_K]: { blockSize: 256, bytesPerBlock: 144 },
  [GGML_TYPE.Q5_K]: { blockSize: 256, bytesPerBlock: 176 },
  [GGML_TYPE.Q6_K]: { blockSize: 256, bytesPerBlock: 210 },
  [GGML_TYPE.Q8_K]: { blockSize: 256, bytesPerBlock: 292 },
};

// ═══════════════════════════════════════════════════════════════════════════
// Float16 ↔ Float32 Conversion
// ═══════════════════════════════════════════════════════════════════════════

function float16ToFloat32(bits) {
  const sign = (bits >> 15) & 0x1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0.0 : 0.0;
    // Denormalized
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1f) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }
  // Normalized
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

// ═══════════════════════════════════════════════════════════════════════════
// GGUF Binary Parser
// ═══════════════════════════════════════════════════════════════════════════

class GGUFParser {
  constructor(buffer) {
    this.buf = buffer;
    this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    this.pos = 0;
  }

  // ── Primitive readers (all little-endian) ──

  readU8()  { const v = this.view.getUint8(this.pos); this.pos += 1; return v; }
  readI8()  { const v = this.view.getInt8(this.pos); this.pos += 1; return v; }
  readU16() { const v = this.view.getUint16(this.pos, true); this.pos += 2; return v; }
  readI16() { const v = this.view.getInt16(this.pos, true); this.pos += 2; return v; }
  readU32() { const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  readI32() { const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; }
  readF32() { const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
  readF64() { const v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; }

  readU64() {
    const lo = this.view.getUint32(this.pos, true);
    const hi = this.view.getUint32(this.pos + 4, true);
    this.pos += 8;
    // For values that fit in Number (< 2^53), return Number. Otherwise BigInt.
    if (hi === 0 && lo >= 0) return lo;
    return BigInt(lo) + (BigInt(hi) << 32n);
  }

  readI64() {
    const lo = this.view.getUint32(this.pos, true);
    const hi = this.view.getInt32(this.pos + 4, true);
    this.pos += 8;
    if (hi === 0 && lo >= 0) return lo;
    if (hi === -1 && lo > 0) return -(0x100000000 - lo);
    return BigInt(lo) + (BigInt(hi) << 32n);
  }

  readBool() { return this.readU8() !== 0; }

  readString() {
    const len = Number(this.readU64());
    const bytes = this.buf.slice(this.pos, this.pos + len);
    this.pos += len;
    return bytes.toString('utf-8');
  }

  // ── Typed value reader ──

  readValue(type) {
    switch (type) {
      case GGUF_TYPE.UINT8:   return this.readU8();
      case GGUF_TYPE.INT8:    return this.readI8();
      case GGUF_TYPE.UINT16:  return this.readU16();
      case GGUF_TYPE.INT16:   return this.readI16();
      case GGUF_TYPE.UINT32:  return this.readU32();
      case GGUF_TYPE.INT32:   return this.readI32();
      case GGUF_TYPE.FLOAT32: return this.readF32();
      case GGUF_TYPE.BOOL:    return this.readBool();
      case GGUF_TYPE.STRING:  return this.readString();
      case GGUF_TYPE.ARRAY:   return this.readArray();
      case GGUF_TYPE.UINT64:  return this.readU64();
      case GGUF_TYPE.INT64:   return this.readI64();
      case GGUF_TYPE.FLOAT64: return this.readF64();
      default: throw new Error(`Unknown GGUF value type: ${type}`);
    }
  }

  readArray() {
    const elemType = this.readU32();
    const count = Number(this.readU64());
    const arr = new Array(count);
    for (let i = 0; i < count; i++) {
      arr[i] = this.readValue(elemType);
    }
    return arr;
  }

  // ── High-level parsing ──

  readHeader() {
    const magic = this.readU32();
    if (magic !== GGUF_MAGIC) {
      throw new Error(`Not a GGUF file (magic: 0x${magic.toString(16)}, expected 0x${GGUF_MAGIC.toString(16)})`);
    }
    const version = this.readU32();
    if (version < 2 || version > 3) {
      throw new Error(`Unsupported GGUF version: ${version} (supports v2-v3)`);
    }
    const tensorCount = Number(this.readU64());
    const metadataKvCount = Number(this.readU64());
    return { version, tensorCount, metadataKvCount };
  }

  readMetadata(count) {
    const metadata = {};
    for (let i = 0; i < count; i++) {
      const key = this.readString();
      const valueType = this.readU32();
      const value = this.readValue(valueType);
      metadata[key] = value;
    }
    return metadata;
  }

  readTensorInfos(count) {
    const infos = [];
    for (let i = 0; i < count; i++) {
      const name = this.readString();
      const nDims = this.readU32();
      const dims = [];
      for (let d = 0; d < nDims; d++) {
        dims.push(Number(this.readU64()));
      }
      const type = this.readU32();
      const offset = Number(this.readU64());
      infos.push({ name, dims, type, offset });
    }
    return infos;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tensor Data Loading — Dequantize to Float32
// ═══════════════════════════════════════════════════════════════════════════

function tensorElementCount(dims) {
  let n = 1;
  for (const d of dims) n *= d;
  return n;
}

function tensorByteSize(dims, type) {
  const n = tensorElementCount(dims);
  const info = GGML_TYPE_SIZE[type];
  if (!info) throw new Error(`Unsupported tensor type: ${type}`);
  if (info.blockSize === 1) return n * info.bytesPerBlock;
  const nBlocks = Math.ceil(n / info.blockSize);
  return nBlocks * info.bytesPerBlock;
}

/**
 * Load tensor data from buffer and convert to Float32Array.
 * Handles F32, F16, Q8_0, Q4_0 dequantization.
 */
function loadTensorData(buffer, offset, dims, type) {
  const n = tensorElementCount(dims);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  if (type === GGML_TYPE.F32) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = view.getFloat32(offset + i * 4, true);
    }
    return out;
  }

  if (type === GGML_TYPE.F16) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = float16ToFloat32(view.getUint16(offset + i * 2, true));
    }
    return out;
  }

  if (type === GGML_TYPE.Q8_0) {
    // Block size 32: [f16 scale (2 bytes)] [32 x int8 (32 bytes)] = 34 bytes per block
    const out = new Float32Array(n);
    const nBlocks = Math.ceil(n / 32);
    let outIdx = 0;
    let blockOffset = offset;

    for (let b = 0; b < nBlocks; b++) {
      const scale = float16ToFloat32(view.getUint16(blockOffset, true));
      blockOffset += 2;

      const remaining = Math.min(32, n - outIdx);
      for (let i = 0; i < remaining; i++) {
        out[outIdx++] = view.getInt8(blockOffset + i) * scale;
      }
      blockOffset += 32;
    }
    return out;
  }

  if (type === GGML_TYPE.Q4_0) {
    // Block size 32: [f16 scale (2 bytes)] [16 bytes (32 x 4-bit)] = 18 bytes per block
    // GGUF nibble layout: low nibbles of bytes 0-15 → values 0-15,
    //                      high nibbles of bytes 0-15 → values 16-31
    // Values are unsigned (0-15), offset by 8: actual = (q - 8) * scale
    const out = new Float32Array(n);
    const nBlocks = Math.ceil(n / 32);
    let outIdx = 0;
    let blockOffset = offset;

    for (let b = 0; b < nBlocks; b++) {
      const scale = float16ToFloat32(view.getUint16(blockOffset, true));
      blockOffset += 2;

      const remaining = Math.min(32, n - outIdx);
      const halfBlock = Math.min(16, remaining);
      for (let j = 0; j < halfBlock; j++) {
        const byte = view.getUint8(blockOffset + j);
        out[outIdx + j] = ((byte & 0x0F) - 8) * scale;
        if (j + 16 < remaining) {
          out[outIdx + j + 16] = ((byte >> 4) - 8) * scale;
        }
      }
      outIdx += remaining;
      blockOffset += 16;
    }
    return out;
  }

  if (type === GGML_TYPE.Q4_1) {
    // Block size 32: [f16 scale (2)] [f16 min (2)] [16 bytes (32 x 4-bit)] = 20 bytes per block
    // Same nibble layout as Q4_0. Values are unsigned (0-15): actual = q * d + m
    const out = new Float32Array(n);
    const nBlocks = Math.ceil(n / 32);
    let outIdx = 0;
    let blockOffset = offset;

    for (let b = 0; b < nBlocks; b++) {
      const d = float16ToFloat32(view.getUint16(blockOffset, true));
      blockOffset += 2;
      const m = float16ToFloat32(view.getUint16(blockOffset, true));
      blockOffset += 2;

      const remaining = Math.min(32, n - outIdx);
      const halfBlock = Math.min(16, remaining);
      for (let j = 0; j < halfBlock; j++) {
        const byte = view.getUint8(blockOffset + j);
        out[outIdx + j] = (byte & 0x0F) * d + m;
        if (j + 16 < remaining) {
          out[outIdx + j + 16] = (byte >> 4) * d + m;
        }
      }
      outIdx += remaining;
      blockOffset += 16;
    }
    return out;
  }

  if (type === GGML_TYPE.Q6_K) {
    // Block size 256: [ql: 128 bytes] [qh: 64 bytes] [scales: 16 bytes] [d: 2 bytes f16] = 210 bytes
    // 6-bit quantization with per-sub-block int8 scales and f16 super-scale
    const out = new Float32Array(n);
    const nBlocks = Math.ceil(n / 256);
    let outIdx = 0;

    for (let b = 0; b < nBlocks; b++) {
      const blockOffset = offset + b * 210;
      const qlOff = blockOffset;        // 128 bytes: low 4 bits
      const qhOff = blockOffset + 128;  //  64 bytes: high 2 bits
      const scOff = blockOffset + 192;  //  16 bytes: int8 scales
      const d = float16ToFloat32(view.getUint16(blockOffset + 208, true));

      // Process two halves of 128 values each
      for (let half = 0; half < 2; half++) {
        const qlBase = qlOff + half * 64;
        const qhBase = qhOff + half * 32;

        for (let l = 0; l < 32; l++) {
          const is = half * 8 + Math.floor(l / 16);

          const qlByte0 = view.getUint8(qlBase + l);
          const qlByte32 = view.getUint8(qlBase + l + 32);
          const qhByte = view.getUint8(qhBase + l);

          // Reconstruct 6-bit values: low 4 bits from ql, high 2 bits from qh
          const q1 = ((qlByte0 & 0xF) | (((qhByte >> 0) & 3) << 4)) - 32;
          const q2 = ((qlByte32 & 0xF) | (((qhByte >> 2) & 3) << 4)) - 32;
          const q3 = ((qlByte0 >> 4) | (((qhByte >> 4) & 3) << 4)) - 32;
          const q4 = ((qlByte32 >> 4) | (((qhByte >> 6) & 3) << 4)) - 32;

          const sc0 = view.getInt8(scOff + is + 0);
          const sc2 = view.getInt8(scOff + is + 2);
          const sc4 = view.getInt8(scOff + is + 4);
          const sc6 = view.getInt8(scOff + is + 6);

          const idx = outIdx + half * 128;
          if (idx + l < n)      out[idx + l]      = d * sc0 * q1;
          if (idx + l + 32 < n) out[idx + l + 32] = d * sc2 * q2;
          if (idx + l + 64 < n) out[idx + l + 64] = d * sc4 * q3;
          if (idx + l + 96 < n) out[idx + l + 96] = d * sc6 * q4;
        }
      }
      outIdx += 256;
    }
    return out;
  }

  throw new Error(`Unsupported tensor type for loading: ${type}`);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tensor Name Mapping — GGUF names → PureBee names
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Map GGUF tensor name to our internal naming convention.
 * Returns { name, transpose } where transpose indicates if the weight
 * needs to be transposed for our matmul convention.
 */
function mapTensorName(ggufName) {
  // Embedding
  if (ggufName === 'token_embd.weight') return { name: 'token_embedding', transpose: false };

  // Output (LM head)
  if (ggufName === 'output.weight') return { name: 'lm_head', transpose: false };

  // Final norm
  if (ggufName === 'output_norm.weight') return { name: 'rms_final', transpose: false };

  // Per-layer patterns
  const layerMatch = ggufName.match(/^blk\.(\d+)\.(.+)$/);
  if (layerMatch) {
    const layer = layerMatch[1];
    const suffix = layerMatch[2];
    const prefix = `layer${layer}`;

    switch (suffix) {
      case 'attn_norm.weight':   return { name: `${prefix}.rms_att`, transpose: false };
      case 'ffn_norm.weight':    return { name: `${prefix}.rms_ffn`, transpose: false };
      case 'attn_q.weight':      return { name: `${prefix}.wq`, transpose: true };
      case 'attn_k.weight':      return { name: `${prefix}.wk`, transpose: true };
      case 'attn_v.weight':      return { name: `${prefix}.wv`, transpose: true };
      case 'attn_output.weight': return { name: `${prefix}.wo`, transpose: true };
      case 'ffn_gate.weight':    return { name: `${prefix}.w1`, transpose: true };
      case 'ffn_down.weight':    return { name: `${prefix}.w2`, transpose: true };
      case 'ffn_up.weight':      return { name: `${prefix}.w3`, transpose: true };
    }
  }

  // Unknown — keep original name, no transpose
  return { name: ggufName, transpose: false };
}

/**
 * Transpose a 2D Float32Array from [rows, cols] to [cols, rows].
 */
function transpose2D(data, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[c * rows + r] = data[r * cols + c];
    }
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Config Extraction
// ═══════════════════════════════════════════════════════════════════════════

function extractConfig(metadata) {
  const arch = metadata['general.architecture'] || 'llama';
  const prefix = arch;

  const config = {
    arch,
    name: metadata['general.name'] || 'unknown',
    dim: metadata[`${prefix}.embedding_length`],
    hiddenDim: metadata[`${prefix}.feed_forward_length`],
    nLayers: metadata[`${prefix}.block_count`],
    nHeads: metadata[`${prefix}.attention.head_count`],
    nKvHeads: metadata[`${prefix}.attention.head_count_kv`] || metadata[`${prefix}.attention.head_count`],
    vocabSize: null, // Set from tokenizer
    seqLen: metadata[`${prefix}.context_length`] || 2048,
    rmsNormEps: metadata[`${prefix}.attention.layer_norm_rms_epsilon`] || 1e-5,
    ropeTheta: metadata[`${prefix}.rope.freq_base`] || 10000.0,
  };

  config.headDim = Math.floor(config.dim / config.nHeads);

  return config;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer Extraction
// ═══════════════════════════════════════════════════════════════════════════

/**
 * GPT-2 byte-to-unicode mapping.
 * Maps all 256 bytes to unique unicode characters.
 * Printable ASCII maps to itself, non-printable maps to U+0100+.
 */
function buildByteToUnicode() {
  const byteToUni = new Array(256);
  const uniToByte = {};

  // Printable ranges that map to themselves
  const ranges = [[33, 126], [161, 172], [174, 255]];
  const selfMap = new Set();
  for (const [lo, hi] of ranges) {
    for (let i = lo; i <= hi; i++) selfMap.add(i);
  }

  let next = 256;
  for (let b = 0; b < 256; b++) {
    if (selfMap.has(b)) {
      byteToUni[b] = String.fromCharCode(b);
    } else {
      byteToUni[b] = String.fromCharCode(next);
      next++;
    }
    uniToByte[byteToUni[b]] = b;
  }

  return { byteToUni, uniToByte };
}

class GGUFTokenizer {
  constructor(metadata) {
    this.model = metadata['tokenizer.ggml.model'] || 'llama';
    this.vocab = metadata['tokenizer.ggml.tokens'] || [];
    this.scores = metadata['tokenizer.ggml.scores'] || [];
    this.tokenTypes = metadata['tokenizer.ggml.token_type'] || [];
    this.bosId = metadata['tokenizer.ggml.bos_token_id'] ?? 1;
    this.eosId = metadata['tokenizer.ggml.eos_token_id'] ?? 2;

    // Build token → ID lookup
    this._tokenToId = new Map();
    for (let i = 0; i < this.vocab.length; i++) {
      this._tokenToId.set(this.vocab[i], i);
    }

    // Build merge pair → rank lookup from GGUF merges list.
    // GPT-2 tokenizers use merges (not scores) for BPE priority.
    // Each merge "A B" at index i has rank i (lower rank = higher priority).
    const mergesList = metadata['tokenizer.ggml.merges'] || [];
    this._mergeRank = new Map(); // "idA,idB" → rank
    for (let i = 0; i < mergesList.length; i++) {
      const spaceIdx = mergesList[i].indexOf(' ');
      if (spaceIdx === -1) continue;
      const a = mergesList[i].slice(0, spaceIdx);
      const b = mergesList[i].slice(spaceIdx + 1);
      const aId = this._tokenToId.get(a);
      const bId = this._tokenToId.get(b);
      if (aId !== undefined && bId !== undefined) {
        this._mergeRank.set(`${aId},${bId}`, i);
      }
    }

    // Build byte ↔ unicode mapping for GPT-2 style tokenizers
    if (this.model === 'gpt2') {
      const { byteToUni, uniToByte } = buildByteToUnicode();
      this._byteToUni = byteToUni;
      this._uniToByte = uniToByte;
    }

    // Build byte token lookup for SentencePiece style
    this._byteTokens = new Map(); // byte value → token ID
    for (let i = 0; i < this.vocab.length; i++) {
      if (this.tokenTypes[i] === 6) { // byte type
        // Byte tokens are like "<0x00>", "<0x01>", etc.
        const match = this.vocab[i].match(/^<0x([0-9A-Fa-f]{2})>$/);
        if (match) {
          this._byteTokens.set(parseInt(match[1], 16), i);
        }
      }
    }

    // Find special tokens for chat templates
    // Register all tokens with <|...|> pattern and control tokens (type 3)
    this._specialTokens = new Map();
    for (let i = 0; i < this.vocab.length; i++) {
      const t = this.vocab[i];
      if (t === '<s>' || t === '</s>') {
        this._specialTokens.set(t, i);
      } else if (t.startsWith('<|') && t.endsWith('|>')) {
        this._specialTokens.set(t, i);
      }
    }
  }

  /**
   * Encode text to token IDs using BPE.
   * Handles special tokens (like <|im_start|>, <|im_end|>) as single tokens.
   */
  encode(text) {
    // Split on special tokens, encode each segment separately
    if (this._specialTokens.size > 0) {
      return this._encodeWithSpecials(text);
    }
    if (this.model === 'gpt2') {
      return this._encodeGPT2(text);
    }
    return this._encodeSentencePiece(text);
  }

  /**
   * Split text on special tokens and encode segments individually.
   */
  _encodeWithSpecials(text) {
    // Build regex to match any special token
    const specials = Array.from(this._specialTokens.keys());
    // Escape regex special chars in token strings, sort by length (longest first)
    const escaped = specials
      .sort((a, b) => b.length - a.length)
      .map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const pattern = new RegExp(`(${escaped.join('|')})`, 'g');

    const tokens = [];
    let lastIdx = 0;

    for (const match of text.matchAll(pattern)) {
      // Encode text before this special token
      const before = text.slice(lastIdx, match.index);
      if (before.length > 0) {
        if (this.model === 'gpt2') {
          tokens.push(...this._encodeGPT2(before));
        } else {
          tokens.push(...this._encodeSentencePiece(before));
        }
      }
      // Insert the special token ID
      tokens.push(this._specialTokens.get(match[0]));
      lastIdx = match.index + match[0].length;
    }

    // Encode remaining text after last special token
    const remaining = text.slice(lastIdx);
    if (remaining.length > 0) {
      if (this.model === 'gpt2') {
        tokens.push(...this._encodeGPT2(remaining));
      } else {
        tokens.push(...this._encodeSentencePiece(remaining));
      }
    }

    return tokens;
  }

  /**
   * GPT-2 style BPE encoding.
   */
  _encodeGPT2(text) {
    // Convert text to byte-level unicode representation
    const textBytes = Buffer.from(text, 'utf-8');
    const uniChars = [];
    for (let i = 0; i < textBytes.length; i++) {
      uniChars.push(this._byteToUni[textBytes[i]]);
    }
    const uniStr = uniChars.join('');

    // Split into initial single-character tokens
    let tokens = [];
    for (const ch of uniStr) {
      const id = this._tokenToId.get(ch);
      if (id !== undefined) {
        tokens.push(id);
      } else {
        // Unknown character — skip (shouldn't happen with byte-level BPE)
        tokens.push(this._tokenToId.get('<unk>') || 0);
      }
    }

    // BPE merge loop: repeatedly merge the highest-scoring pair
    tokens = this._bpeMerge(tokens);
    return tokens;
  }

  /**
   * SentencePiece style BPE encoding.
   */
  _encodeSentencePiece(text) {
    // SentencePiece: replace leading/between-word spaces with ▁
    const processed = text.replace(/ /g, '\u2581');
    // If text doesn't start with space, add ▁ prefix (SentencePiece convention)
    const spText = (text.length > 0 && text[0] !== ' ') ? '\u2581' + processed : processed;

    // Convert to initial tokens (try single characters, fall back to byte tokens)
    let tokens = [];
    for (const ch of spText) {
      const id = this._tokenToId.get(ch);
      if (id !== undefined) {
        tokens.push(id);
      } else {
        // Byte fallback
        const bytes = Buffer.from(ch, 'utf-8');
        for (const b of bytes) {
          const byteId = this._byteTokens.get(b);
          if (byteId !== undefined) {
            tokens.push(byteId);
          }
        }
      }
    }

    // BPE merge loop
    tokens = this._bpeMerge(tokens);
    return tokens;
  }

  /**
   * BPE merge loop — repeatedly merge the lowest-rank (highest priority) adjacent pair.
   * Uses the merges list for GPT-2 tokenizers, or scores for SentencePiece.
   */
  _bpeMerge(tokens) {
    if (tokens.length < 2) return tokens;

    while (true) {
      let bestRank = Infinity;
      let bestIdx = -1;
      let bestMergedId = -1;

      for (let i = 0; i < tokens.length - 1; i++) {
        const key = `${tokens[i]},${tokens[i + 1]}`;
        const rank = this._mergeRank.get(key);
        if (rank !== undefined && rank < bestRank) {
          // Verify the merged token exists in vocabulary
          const merged = this.vocab[tokens[i]] + this.vocab[tokens[i + 1]];
          const mergedId = this._tokenToId.get(merged);
          if (mergedId !== undefined) {
            bestRank = rank;
            bestIdx = i;
            bestMergedId = mergedId;
          }
        }
      }

      if (bestIdx === -1) break; // No more merges possible

      // Apply the merge
      tokens.splice(bestIdx, 2, bestMergedId);
    }

    return tokens;
  }

  /**
   * Decode token IDs to text.
   */
  decode(tokenIds) {
    let text = '';
    for (let i = 0; i < tokenIds.length; i++) {
      text += this._decodeToken(tokenIds[i]);
    }
    return text;
  }

  /**
   * Decode a single token to text (for streaming output).
   */
  decodeToken(tokenId, prevTokenId) {
    return this._decodeToken(tokenId);
  }

  _decodeToken(tokenId) {
    if (tokenId < 0 || tokenId >= this.vocab.length) return '';
    const token = this.vocab[tokenId];

    // Skip control tokens in output
    if (this.tokenTypes[tokenId] === 3) return ''; // control token

    if (this.model === 'gpt2') {
      // Convert GPT-2 unicode back to bytes
      const bytes = [];
      for (const ch of token) {
        const b = this._uniToByte[ch];
        if (b !== undefined) {
          bytes.push(b);
        }
      }
      return Buffer.from(bytes).toString('utf-8');
    }

    // SentencePiece: replace ▁ with space
    if (token.startsWith('\u2581')) {
      return ' ' + token.slice(1);
    }

    // Byte token: <0xNN>
    const byteMatch = token.match(/^<0x([0-9A-Fa-f]{2})>$/);
    if (byteMatch) {
      return String.fromCharCode(parseInt(byteMatch[1], 16));
    }

    return token;
  }

  /**
   * Get special token ID by name.
   */
  getSpecialToken(name) {
    return this._specialTokens.get(name);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Loader — Parse GGUF and return PureBee-compatible model
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Load a GGUF model file and return it in PureBee-compatible format.
 *
 * @param {string} filepath - Path to .gguf file
 * @returns {{ config, weights, tokenizer, sharedWeights }}
 *   config: { dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, seqLen, headDim }
 *   weights: { name: Float32Array } — transposed for PureBee matmul convention
 *   tokenizer: GGUFTokenizer instance
 *   sharedWeights: boolean
 */
function loadGGUFModel(filepath) {
  console.log(`  [GGUF] Reading ${filepath}...`);
  const buffer = fs.readFileSync(filepath);
  console.log(`  [GGUF] File size: ${(buffer.length / 1024 / 1024).toFixed(1)}MB`);

  const parser = new GGUFParser(buffer);

  // ── Parse header and metadata ──
  const header = parser.readHeader();
  console.log(`  [GGUF] Version: ${header.version}, ${header.tensorCount} tensors, ${header.metadataKvCount} metadata entries`);

  const metadata = parser.readMetadata(header.metadataKvCount);
  const tensorInfos = parser.readTensorInfos(header.tensorCount);

  // ── Extract config ──
  const config = extractConfig(metadata);
  console.log(`  [GGUF] Architecture: ${config.arch}, ${config.name}`);
  console.log(`  [GGUF] Config: dim=${config.dim}, hidden=${config.hiddenDim}, layers=${config.nLayers}, heads=${config.nHeads}, kv_heads=${config.nKvHeads}, seq=${config.seqLen}`);

  // ── Extract tokenizer ──
  const tokenizer = new GGUFTokenizer(metadata);
  config.vocabSize = tokenizer.vocab.length;
  console.log(`  [GGUF] Tokenizer: ${tokenizer.model} model, ${config.vocabSize} tokens, BOS=${tokenizer.bosId}, EOS=${tokenizer.eosId}`);

  // ── Calculate tensor data start ──
  const alignment = metadata['general.alignment'] || 32;
  const dataStart = Math.ceil(parser.pos / alignment) * alignment;

  // ── Load and convert tensors ──
  const weights = {};
  let totalParams = 0;
  let hasOutputWeight = false;

  for (const info of tensorInfos) {
    const { name: tensorName, transpose } = mapTensorName(info.name);

    // Load tensor data as Float32Array
    const data = loadTensorData(buffer, dataStart + info.offset, info.dims, info.type);
    const nElements = tensorElementCount(info.dims);
    totalParams += nElements;

    if (info.name === 'output.weight') hasOutputWeight = true;

    // Transpose 2D weight matrices for our matmul convention.
    // GGUF dims: [cols, rows] — data is stored as rows×cols in row-major.
    // Our matmul expects W[K,N] = [in_dim, out_dim], so we transpose from
    // the GGUF layout [out_dim, in_dim] to [in_dim, out_dim].
    if (transpose && info.dims.length === 2) {
      const rows = info.dims[1]; // actual rows in memory (out_features)
      const cols = info.dims[0]; // actual cols in memory (in_features)
      weights[tensorName] = transpose2D(data, rows, cols);
    } else {
      weights[tensorName] = data;
    }
  }

  const sharedWeights = !hasOutputWeight;
  console.log(`  [GGUF] ${tensorInfos.length} tensors, ${(totalParams / 1e6).toFixed(1)}M parameters`);
  console.log(`  [GGUF] Shared weights: ${sharedWeights}`);

  return { config, weights, tokenizer, sharedWeights };
}

module.exports = {
  loadGGUFModel, GGUFTokenizer, GGUFParser, GGML_TYPE, GGML_TYPE_SIZE,
  loadTensorData, tensorByteSize, tensorElementCount,
  mapTensorName, transpose2D, extractConfig, float16ToFloat32,
};
