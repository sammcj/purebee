/**
 * PureBee — 3 — Quantization Engine
 *
 * Quantization as a first-class citizen, not a compression hack.
 * Real GPUs dequantize to float32 then compute. PureBee operates on
 * quantized values directly where possible, decompressing only when needed.
 *
 * Supported formats:
 *   Q8_0 — 8-bit integers, block size 32, 1 float32 scale per block
 *          Memory: ~4.25x smaller than float32
 *          Speed:  Faster due to reduced memory bandwidth
 *
 *   Q4_0 — 4-bit integers, block size 32, 1 float32 scale per block
 *          Memory: ~8.5x smaller than float32
 *          Speed:  Even faster, slight quality loss
 *
 * Zero external dependencies.
 */

'use strict';

const BLOCK_SIZE = 32;

/**
 * A quantized tensor — stores weights as scaled integers.
 *
 * Each block of 32 values shares a single scale factor.
 * Original value ≈ int_value * scale
 */
class QuantizedTensor {
  /**
   * @param {string} name
   * @param {number[]} shape - original tensor shape
   * @param {string} type - 'q8_0' or 'q4_0'
   */
  constructor(name, shape, type = 'q8_0') {
    this.name = name;
    this.shape = shape;
    this.type = type;
    this.size = shape.reduce((a, b) => a * b, 1);
    this.numBlocks = Math.ceil(this.size / BLOCK_SIZE);
    this.scales = new Float32Array(this.numBlocks);

    if (type === 'q8_0') {
      this.data = new Int8Array(this.numBlocks * BLOCK_SIZE);
    } else if (type === 'q4_0') {
      // 4-bit: pack 2 values per byte
      this.data = new Uint8Array(Math.ceil(this.numBlocks * BLOCK_SIZE / 2));
    }
  }

  get bytes() {
    return this.scales.byteLength + this.data.byteLength;
  }

  get compressionRatio() {
    const originalBytes = this.size * 4; // float32
    return (originalBytes / this.bytes).toFixed(1);
  }

  toString() {
    const mb = (this.bytes / 1024 / 1024).toFixed(2);
    return `QTensor(${this.name}, ${this.type}, shape=[${this.shape}], ${mb}MB, ${this.compressionRatio}x compression)`;
  }
}

/**
 * Quantize a Float32Array to Q8_0 format.
 *
 * For each block of 32 values:
 *   scale = max(|values|) / 127
 *   quantized[i] = round(value[i] / scale)
 *
 * @param {string} name
 * @param {number[]} shape
 * @param {Float32Array} data
 * @returns {QuantizedTensor}
 */
function quantize_q8(name, shape, data) {
  const qt = new QuantizedTensor(name, shape, 'q8_0');
  const size = data.length;

  for (let b = 0; b < qt.numBlocks; b++) {
    const offset = b * BLOCK_SIZE;
    const end = Math.min(offset + BLOCK_SIZE, size);

    // Find max absolute value in block
    let maxAbs = 0;
    for (let i = offset; i < end; i++) {
      const abs = Math.abs(data[i]);
      if (abs > maxAbs) maxAbs = abs;
    }

    // Compute scale
    const scale = maxAbs / 127;
    qt.scales[b] = scale;

    // Quantize
    if (scale > 0) {
      const invScale = 127 / maxAbs;
      for (let i = offset; i < end; i++) {
        qt.data[i] = Math.round(data[i] * invScale);
      }
    }
    // else: all zeros, data already initialized to 0
  }

  return qt;
}

/**
 * Quantize a Float32Array to Q4_0 format.
 *
 * For each block of 32 values:
 *   scale = max(|values|) / 7
 *   quantized[i] = round(value[i] / scale)  // range [-8, 7]
 *   Packed: two 4-bit values per byte (low nibble first)
 *
 * @param {string} name
 * @param {number[]} shape
 * @param {Float32Array} data
 * @returns {QuantizedTensor}
 */
function quantize_q4(name, shape, data) {
  const qt = new QuantizedTensor(name, shape, 'q4_0');
  const size = data.length;

  for (let b = 0; b < qt.numBlocks; b++) {
    const offset = b * BLOCK_SIZE;
    const end = Math.min(offset + BLOCK_SIZE, size);

    // Find max absolute value
    let maxAbs = 0;
    for (let i = offset; i < end; i++) {
      const abs = Math.abs(data[i]);
      if (abs > maxAbs) maxAbs = abs;
    }

    // Scale to [-8, 7] range (4-bit signed)
    const scale = maxAbs / 7;
    qt.scales[b] = scale;

    if (scale > 0) {
      const invScale = 7 / maxAbs;
      for (let i = offset; i < end; i++) {
        let q = Math.round(data[i] * invScale);
        q = Math.max(-8, Math.min(7, q));
        // Pack: store as unsigned offset by 8 (range 0-15)
        const unsigned = q + 8;
        const byteIdx = Math.floor(i / 2);
        if (i % 2 === 0) {
          qt.data[byteIdx] = (qt.data[byteIdx] & 0xF0) | (unsigned & 0x0F);
        } else {
          qt.data[byteIdx] = (qt.data[byteIdx] & 0x0F) | ((unsigned & 0x0F) << 4);
        }
      }
    }
  }

  return qt;
}

/**
 * Dequantize a Q8_0 tensor back to Float32Array.
 * Used for verification and fallback operations.
 *
 * @param {QuantizedTensor} qt
 * @returns {Float32Array}
 */
function dequantize_q8(qt) {
  const out = new Float32Array(qt.size);
  for (let b = 0; b < qt.numBlocks; b++) {
    const offset = b * BLOCK_SIZE;
    const scale = qt.scales[b];
    const end = Math.min(offset + BLOCK_SIZE, qt.size);
    for (let i = offset; i < end; i++) {
      out[i] = qt.data[i] * scale;
    }
  }
  return out;
}

/**
 * Dequantize a Q4_0 tensor back to Float32Array.
 *
 * @param {QuantizedTensor} qt
 * @returns {Float32Array}
 */
function dequantize_q4(qt) {
  const out = new Float32Array(qt.size);
  for (let b = 0; b < qt.numBlocks; b++) {
    const offset = b * BLOCK_SIZE;
    const scale = qt.scales[b];
    const end = Math.min(offset + BLOCK_SIZE, qt.size);
    for (let i = offset; i < end; i++) {
      const byteIdx = Math.floor(i / 2);
      let nibble;
      if (i % 2 === 0) {
        nibble = qt.data[byteIdx] & 0x0F;
      } else {
        nibble = (qt.data[byteIdx] >> 4) & 0x0F;
      }
      // Convert from unsigned [0,15] back to signed [-8, 7]
      out[i] = (nibble - 8) * scale;
    }
  }
  return out;
}

/**
 * Quantized matrix multiplication: C = A (float32) @ B (quantized)
 *
 * This is the key operation. Instead of loading 4 bytes per weight,
 * we load 1 byte (Q8) or 0.5 bytes (Q4) and dequantize on the fly.
 * Less memory bandwidth = faster on memory-bound operations.
 *
 * @param {Float32Array} aData - input activation [M, K]
 * @param {number} M
 * @param {number} K
 * @param {QuantizedTensor} B_q - quantized weight [K, N]
 * @param {Float32Array|null} biasData - optional bias [N]
 * @param {Float32Array} outData - output buffer [M, N]
 */
function matmul_q8(aData, M, K, B_q, biasData, outData) {
  const N = B_q.shape[1];
  const scales = B_q.scales;
  const qData = B_q.data;
  const blocksPerRow = Math.ceil(N / BLOCK_SIZE);

  // Initialize output with bias or zeros
  for (let m = 0; m < M; m++) {
    const outBase = m * N;
    if (biasData) {
      for (let n = 0; n < N; n++) outData[outBase + n] = biasData[n];
    } else {
      for (let n = 0; n < N; n++) outData[outBase + n] = 0;
    }
  }

  // Cache-friendly loop order: m, k, n_block
  // Process B in blocks of BLOCK_SIZE along the N dimension
  for (let m = 0; m < M; m++) {
    const aBase = m * K;
    const outBase = m * N;

    for (let k = 0; k < K; k++) {
      const aVal = aData[aBase + k];
      if (aVal === 0) continue; // Skip zero activations (sparse execution!)

      const bRowOffset = k * N;

      // Process by blocks for scale reuse
      for (let blockN = 0; blockN < blocksPerRow; blockN++) {
        const nStart = blockN * BLOCK_SIZE;
        const nEnd = Math.min(nStart + BLOCK_SIZE, N);
        const globalBlockIdx = Math.floor((bRowOffset + nStart) / BLOCK_SIZE);
        const scale = scales[globalBlockIdx];
        const aScaled = aVal * scale;

        for (let n = nStart; n < nEnd; n++) {
          outData[outBase + n] += aScaled * qData[bRowOffset + n];
        }
      }
    }
  }
}

/**
 * Quantize all weight tensors in a model's weight dictionary.
 *
 * @param {Object} weights - { name: Float32Array }
 * @param {Object} shapes - { name: [shape] } — original shapes
 * @param {string} type - 'q8_0' or 'q4_0'
 * @returns {{ weights: Object, originalMB: number, quantizedMB: number }}
 */
function quantizeWeights(weights, shapes, type = 'q8_0') {
  const quantFn = type === 'q4_0' ? quantize_q4 : quantize_q8;
  const quantized = {};
  let originalBytes = 0;
  let quantizedBytes = 0;

  for (const [name, data] of Object.entries(weights)) {
    const shape = shapes[name];
    if (!shape) {
      // Keep unquantized (e.g., norm weights, biases)
      quantized[name] = data;
      originalBytes += data.byteLength;
      quantizedBytes += data.byteLength;
      continue;
    }

    // Only quantize large weight matrices (skip small norm/bias tensors)
    if (data.length < 256) {
      quantized[name] = data;
      originalBytes += data.byteLength;
      quantizedBytes += data.byteLength;
      continue;
    }

    const qt = quantFn(name, shape, data);
    quantized[name] = qt;
    originalBytes += data.byteLength;
    quantizedBytes += qt.bytes;
  }

  return {
    weights: quantized,
    originalMB: (originalBytes / 1024 / 1024).toFixed(1),
    quantizedMB: (quantizedBytes / 1024 / 1024).toFixed(1),
    ratio: (originalBytes / quantizedBytes).toFixed(1),
  };
}

module.exports = {
  QuantizedTensor,
  quantize_q8,
  quantize_q4,
  dequantize_q8,
  dequantize_q4,
  matmul_q8,
  quantizeWeights,
  BLOCK_SIZE,
};
