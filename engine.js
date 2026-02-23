/**
 * PureBee — Execution Engine
 * 
 * The parallel core. Takes tensors and rules, produces results.
 * This is the heart of PureBee — where math actually runs.
 * 
 * Every operation here maps directly to a GPU shader in spirit:
 * a function applied across a data space, as fast as possible.
 */

'use strict';

const { Tensor } = require('./memory');
const { QuantizedTensor, matmul_q8 } = require('./quantize');
const wasmSimd = require('./wasm-simd');

class ExecutionEngine {
  constructor() {
    this._opCount = 0;
    this._flops = 0;
    this._wasmReady = false;
  }

  /**
   * Initialize WASM SIMD acceleration.
   * Call once at startup. Returns true if WASM SIMD is available.
   */
  initWasm() {
    this._wasmReady = wasmSimd.init();
    return this._wasmReady;
  }

  /**
   * Cache a weight tensor's data in WASM memory for fast matmul.
   * Call during model loading for each weight matrix.
   *
   * @param {Tensor} tensor — weight tensor to cache
   */
  cacheWeight(tensor) {
    if (!this._wasmReady) return;
    tensor._wasmPtr = wasmSimd.allocWeights(tensor.data);
  }

  get wasmStats() {
    return this._wasmReady ? wasmSimd.getStats() : null;
  }

  /**
   * PARALLEL_MAP — apply fn to every element simultaneously.
   * This is the fundamental primitive. Everything else is built on this.
   * 
   * @param {Tensor} input
   * @param {function} fn  (value, index) => new_value
   * @param {string} outName
   * @returns {Tensor}
   */
  parallelMap(input, fn, outName) {
    const out = new Tensor(outName || input.name + '_mapped', [...input.shape]);
    const data = input.data;
    const outData = out.data;
    for (let i = 0; i < data.length; i++) {
      outData[i] = fn(data[i], i);
    }
    this._opCount++;
    this._flops += data.length;
    return out;
  }

  /**
   * PARALLEL_REDUCE — aggregate all values into a scalar.
   * 
   * @param {Tensor} input
   * @param {function} fn  (accumulator, value) => accumulator
   * @param {number} initial
   * @returns {number}
   */
  parallelReduce(input, fn, initial = 0) {
    let acc = initial;
    const data = input.data;
    for (let i = 0; i < data.length; i++) {
      acc = fn(acc, data[i]);
    }
    this._opCount++;
    this._flops += data.length;
    return acc;
  }

  /**
   * TENSOR_ADD — element-wise addition of two tensors.
   * 
   * @param {Tensor} a
   * @param {Tensor} b
   * @param {string} outName
   * @returns {Tensor}
   */
  tensorAdd(a, b, outName) {
    if (a.size !== b.size) throw new Error(`TENSOR_ADD: size mismatch ${a.size} vs ${b.size}`);
    const out = new Tensor(outName || 'add_out', [...a.shape]);
    const aData = a.data, bData = b.data, outData = out.data;
    for (let i = 0; i < aData.length; i++) {
      outData[i] = aData[i] + bData[i];
    }
    this._opCount++;
    this._flops += aData.length;
    return out;
  }

  /**
   * TENSOR_MUL — matrix multiplication.
   * The most important operation. This is 90% of LLM inference.
   * 
   * A: [M, K]  x  B: [K, N]  =>  C: [M, N]
   * 
   * @param {Tensor} A  shape [M, K]
   * @param {Tensor} B  shape [K, N]
   * @param {string} outName
   * @returns {Tensor} shape [M, N]
   */
  tensorMul(A, B, outName) {
    const [M, K] = A.shape;
    const [K2, N] = B.shape;
    if (K !== K2) throw new Error(`TENSOR_MUL: dimension mismatch A[${M},${K}] x B[${K2},${N}]`);

    const out = new Tensor(outName || 'matmul_out', [M, N]);

    // WASM SIMD fast path
    if (this._wasmReady) {
      if (B._wasmPtr !== undefined) {
        wasmSimd.matmulCached(A.data, M, K, B._wasmPtr, N, null, out.data);
      } else {
        wasmSimd.matmul_general(A.data, M, K, B.data, N, out.data);
      }
      this._opCount++;
      this._flops += M * N * K * 2;
      return out;
    }

    // JS fallback: tiled matmul
    const aData = A.data, bData = B.data, outData = out.data;
    const TILE = 64;
    outData.fill(0);

    for (let m0 = 0; m0 < M; m0 += TILE) {
      const mEnd = Math.min(m0 + TILE, M);
      for (let k0 = 0; k0 < K; k0 += TILE) {
        const kEnd = Math.min(k0 + TILE, K);
        for (let n0 = 0; n0 < N; n0 += TILE) {
          const nEnd = Math.min(n0 + TILE, N);
          for (let m = m0; m < mEnd; m++) {
            const aBase = m * K;
            const cBase = m * N;
            for (let k = k0; k < kEnd; k++) {
              const aVal = aData[aBase + k];
              const bBase = k * N;
              for (let n = n0; n < nEnd; n++) {
                outData[cBase + n] += aVal * bData[bBase + n];
              }
            }
          }
        }
      }
    }

    this._opCount++;
    this._flops += M * N * K * 2;
    return out;
  }

  /**
   * TENSOR_MUL_ADD — fused matrix multiply + bias add.
   * y = x @ W + b
   * Very common in transformer layers. Fusing saves a pass over memory.
   * 
   * @param {Tensor} x      [M, K]
   * @param {Tensor} W      [K, N]
   * @param {Tensor|null} b [N] or null
   * @param {string} outName
   * @returns {Tensor} [M, N]
   */
  tensorMulAdd(x, W, b, outName) {
    const [M, K] = x.shape;
    const [K2, N] = W.shape;
    if (K !== K2) throw new Error(`TENSOR_MUL_ADD: dimension mismatch`);

    const out = new Tensor(outName || 'linear_out', [M, N]);

    // WASM SIMD fast path
    if (this._wasmReady) {
      if (W._wasmPtr !== undefined) {
        wasmSimd.matmulCached(x.data, M, K, W._wasmPtr, N, b ? b.data : null, out.data);
      } else {
        // General WASM path: matmul then add bias
        wasmSimd.matmul_general(x.data, M, K, W.data, N, out.data);
        if (b) {
          const outData = out.data, bData = b.data;
          for (let m = 0; m < M; m++) {
            const base = m * N;
            for (let n = 0; n < N; n++) outData[base + n] += bData[n];
          }
        }
      }
      this._opCount++;
      this._flops += M * N * K * 2 + (b ? M * N : 0);
      return out;
    }

    // JS fallback: tiled matmul + bias
    const xData = x.data, wData = W.data, outData = out.data;

    for (let m = 0; m < M; m++) {
      const outOffset = m * N;
      if (b) {
        for (let n = 0; n < N; n++) outData[outOffset + n] = b.data[n];
      } else {
        for (let n = 0; n < N; n++) outData[outOffset + n] = 0;
      }
    }

    const TILE = 64;
    for (let m0 = 0; m0 < M; m0 += TILE) {
      const mEnd = Math.min(m0 + TILE, M);
      for (let k0 = 0; k0 < K; k0 += TILE) {
        const kEnd = Math.min(k0 + TILE, K);
        for (let n0 = 0; n0 < N; n0 += TILE) {
          const nEnd = Math.min(n0 + TILE, N);
          for (let m = m0; m < mEnd; m++) {
            const xBase = m * K;
            const outBase = m * N;
            for (let k = k0; k < kEnd; k++) {
              const xVal = xData[xBase + k];
              const wBase = k * N;
              for (let n = n0; n < nEnd; n++) {
                outData[outBase + n] += xVal * wData[wBase + n];
              }
            }
          }
        }
      }
    }

    this._opCount++;
    this._flops += M * N * K * 2 + (b ? M * N : 0);
    return out;
  }

  /**
   * TENSOR_MUL_ADD_Q8 — fused quantized matrix multiply + bias add.
   * y = x @ W_q8 + b
   *
   * W is a QuantizedTensor (Q8_0). Dequantization happens on-the-fly
   * during multiplication. This uses less memory bandwidth than float32.
   *
   * @param {Tensor} x      [M, K]
   * @param {QuantizedTensor} W_q  quantized [K, N]
   * @param {Tensor|null} b [N] or null
   * @param {string} outName
   * @returns {Tensor} [M, N]
   */
  tensorMulAddQ8(x, W_q, b, outName) {
    const [M, K] = x.shape;
    const N = W_q.shape[1];

    const out = new Tensor(outName || 'linear_q8_out', [M, N]);
    matmul_q8(x.data, M, K, W_q, b ? b.data : null, out.data);

    this._opCount++;
    this._flops += M * N * K * 2 + (b ? M * N : 0);
    return out;
  }

  /**
   * SOFTMAX — normalize a vector into a probability distribution.
   * Applied along the last dimension.
   * 
   * @param {Tensor} input  [M, N] — applies softmax over each row
   * @param {string} outName
   * @returns {Tensor}
   */
  softmax(input, outName) {
    const out = new Tensor(outName || 'softmax_out', [...input.shape]);
    const rows = input.shape[0];
    const cols = input.shape.length > 1 ? input.shape[1] : input.size;
    const iData = input.data, oData = out.data;

    for (let r = 0; r < rows; r++) {
      const offset = r * cols;
      // Find max for numerical stability
      let maxVal = -Infinity;
      for (let c = 0; c < cols; c++) maxVal = Math.max(maxVal, iData[offset + c]);
      // Compute exp(x - max) and sum
      let sum = 0;
      for (let c = 0; c < cols; c++) {
        oData[offset + c] = Math.exp(iData[offset + c] - maxVal);
        sum += oData[offset + c];
      }
      // Normalize
      for (let c = 0; c < cols; c++) oData[offset + c] /= sum;
    }

    this._opCount++;
    this._flops += input.size * 4;
    return out;
  }

  /**
   * LAYER_NORM — normalize across the last dimension.
   * Critical for transformer stability.
   * 
   * @param {Tensor} x       [seq, dim]
   * @param {Tensor} weight  [dim]
   * @param {Tensor} bias    [dim]
   * @param {number} eps
   * @returns {Tensor}
   */
  layerNorm(x, weight, bias, eps = 1e-5, outName) {
    const [seq, dim] = x.shape;
    const out = new Tensor(outName || 'layernorm_out', [seq, dim]);
    const xData = x.data, wData = weight.data, bData = bias ? bias.data : null;
    const outData = out.data;

    for (let s = 0; s < seq; s++) {
      const offset = s * dim;
      // Mean
      let mean = 0;
      for (let d = 0; d < dim; d++) mean += xData[offset + d];
      mean /= dim;
      // Variance
      let variance = 0;
      for (let d = 0; d < dim; d++) {
        const diff = xData[offset + d] - mean;
        variance += diff * diff;
      }
      variance /= dim;
      const std = Math.sqrt(variance + eps);
      // Normalize and scale
      for (let d = 0; d < dim; d++) {
        outData[offset + d] = ((xData[offset + d] - mean) / std) * wData[d] + (bData ? bData[d] : 0);
      }
    }

    this._opCount++;
    this._flops += seq * dim * 8;
    return out;
  }

  /**
   * GELU — Gaussian Error Linear Unit activation.
   * Used in GPT-style transformers instead of ReLU.
   * Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   * 
   * @param {Tensor} input
   * @returns {Tensor}
   */
  gelu(input, outName) {
    const sqrt2overPi = Math.sqrt(2 / Math.PI);
    return this.parallelMap(input, (x) => {
      const inner = sqrt2overPi * (x + 0.044715 * x * x * x);
      return 0.5 * x * (1 + Math.tanh(inner));
    }, outName || 'gelu_out');
  }

  /**
   * ATTENTION — scaled dot-product attention.
   * The core of the transformer. Q, K, V matrices in, attended output out.
   * 
   * scores = softmax(Q @ K^T / sqrt(d_k)) @ V
   * 
   * @param {Tensor} Q  [seq, d_k]
   * @param {Tensor} K  [seq, d_k]
   * @param {Tensor} V  [seq, d_v]
   * @param {boolean} causal  mask future positions
   * @returns {Tensor} [seq, d_v]
   */
  attention(Q, K, V, causal = true, outName) {
    const [seqQ, dK] = Q.shape;
    const [seqK] = K.shape;
    const dV = V.shape[1];
    const scale = 1.0 / Math.sqrt(dK);

    // scores = Q @ K^T  → [seqQ, seqK]
    const scores = new Tensor('attn_scores', [seqQ, seqK]);
    for (let i = 0; i < seqQ; i++) {
      for (let j = 0; j < seqK; j++) {
        // Causal mask — can't attend to future tokens
        if (causal && j > i) {
          scores.data[i * seqK + j] = -1e9;
          continue;
        }
        let dot = 0;
        for (let k = 0; k < dK; k++) {
          dot += Q.data[i * dK + k] * K.data[j * dK + k];
        }
        scores.data[i * seqK + j] = dot * scale;
      }
    }

    // attn_weights = softmax(scores)  → [seqQ, seqK]
    const weights = this.softmax(scores, 'attn_weights');

    // out = weights @ V  → [seqQ, dV]
    const out = new Tensor(outName || 'attn_out', [seqQ, dV]);
    for (let i = 0; i < seqQ; i++) {
      for (let d = 0; d < dV; d++) {
        let sum = 0;
        for (let j = 0; j < seqK; j++) {
          sum += weights.data[i * seqK + j] * V.data[j * dV + d];
        }
        out.data[i * dV + d] = sum;
      }
    }

    this._flops += seqQ * seqK * dK * 2 + seqQ * seqK * dV * 2;
    this._opCount++;
    return out;
  }

  /**
   * RMS_NORM — Root Mean Square normalization.
   * Used in LLaMA instead of LayerNorm. Simpler — no bias, no mean subtraction.
   *
   * rms = sqrt(mean(x^2) + eps)
   * output = (x / rms) * weight
   *
   * @param {Tensor} x       [seq, dim]
   * @param {Tensor} weight  [dim]
   * @param {number} eps
   * @returns {Tensor}
   */
  rmsNorm(x, weight, eps = 1e-5, outName) {
    const [seq, dim] = x.shape;
    const out = new Tensor(outName || 'rmsnorm_out', [seq, dim]);
    const xData = x.data, wData = weight.data, outData = out.data;

    for (let s = 0; s < seq; s++) {
      const offset = s * dim;
      let sumSq = 0;
      for (let d = 0; d < dim; d++) {
        sumSq += xData[offset + d] * xData[offset + d];
      }
      const rms = Math.sqrt(sumSq / dim + eps);
      for (let d = 0; d < dim; d++) {
        outData[offset + d] = (xData[offset + d] / rms) * wData[d];
      }
    }

    this._opCount++;
    this._flops += seq * dim * 5;
    return out;
  }

  /**
   * SILU — Sigmoid Linear Unit activation.
   * Used in LLaMA instead of GELU.
   * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
   *
   * @param {Tensor} input
   * @returns {Tensor}
   */
  silu(input, outName) {
    return this.parallelMap(input, (x) => {
      return x / (1 + Math.exp(-x));
    }, outName || 'silu_out');
  }

  /**
   * ELEMENT_MUL — element-wise multiplication of two tensors.
   * Used in SwiGLU: silu(gate) * up
   *
   * @param {Tensor} a
   * @param {Tensor} b
   * @param {string} outName
   * @returns {Tensor}
   */
  elementMul(a, b, outName) {
    if (a.size !== b.size) throw new Error(`ELEMENT_MUL: size mismatch ${a.size} vs ${b.size}`);
    const out = new Tensor(outName || 'emul_out', [...a.shape]);
    const aData = a.data, bData = b.data, outData = out.data;
    for (let i = 0; i < aData.length; i++) {
      outData[i] = aData[i] * bData[i];
    }
    this._opCount++;
    this._flops += aData.length;
    return out;
  }

  get stats() {
    return { ops: this._opCount, flops: this._flops };
  }

  resetStats() {
    this._opCount = 0;
    this._flops = 0;
  }
}

module.exports = { ExecutionEngine };
