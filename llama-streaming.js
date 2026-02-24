/**
 * PureBee — 6 — Streaming LLaMA Runtime
 *
 * A LLaMA transformer that loads weights one layer at a time from disk.
 * Enables running models far larger than available RAM.
 *
 * Key innovations:
 *   - Layer streaming: only 1 layer's weights resident at a time
 *   - Early exit: stop processing layers when prediction confidence is high
 *   - WASM weight cycling: resetWeights() between layers to reuse WASM memory
 *   - KV-cache rollback: support for speculative decoding rejection
 *
 * Memory budget (Llama 3.2 1B):
 *   Token embeddings: ~1.0 GB (resident)
 *   1 layer weights:  ~280 MB (streamed)
 *   KV-cache:         ~256 MB (persistent)
 *   Total:            ~1.8 GB peak
 *
 * Zero external dependencies. Pure PureBee instructions.
 */

'use strict';

const { PureBee } = require('./purebee');
const { Tensor } = require('./memory');
const wasmSimd = require('./wasm-simd');
const wasmQ4 = require('./wasm-q4');
const { GGML_TYPE } = require('./gguf');
const { AtomicsThreadPool } = require('./thread-pool');

// ═══════════════════════════════════════════════════════════════════════════
// Q4_0 / Q4_1 Matrix-Vector Multiply
//
// Computes y = x @ W^T directly on quantized data, skipping dequantization
// and transpose entirely. Each row of W (one output neuron) is stored as
// consecutive Q4_0/Q4_1 blocks — we iterate row-by-row.
//
// Q4_0 block layout (18 bytes per 32 values):
//   [f16 scale (2 bytes)] [16 bytes: low nibbles → values 0-15,
//                                    high nibbles → values 16-31]
//   actual_value = (nibble - 8) * scale
//
// Q4_1 block layout (20 bytes per 32 values):
//   [f16 delta/scale (2)] [f16 min (2)] [16 bytes nibbles]
//   actual_value = nibble * delta + min
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Inline float16 → float32 conversion (avoids function call overhead).
 */
function f16(bits) {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1f) return frac ? NaN : (sign ? -Infinity : Infinity);
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * Q4_0 matrix-vector multiply: y[N] = W_q4[N, K] @ x[K]
 *
 * @param {Float32Array} x — input vector [K]
 * @param {Buffer|Uint8Array} rawBuf — raw Q4_0 data, N rows × (K/32 × 18) bytes
 * @param {number} N — output dimension (rows)
 * @param {number} K — input dimension (columns, must be multiple of 32)
 * @returns {Float32Array} — output [N]
 */
function q4_0_matvec(x, rawBuf, N, K) {
  const output = new Float32Array(N);
  const blocksPerRow = K >>> 5;  // K / 32
  const bytesPerRow = blocksPerRow * 18;
  const raw = rawBuf instanceof Uint8Array ? rawBuf :
    new Uint8Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength);

  for (let n = 0; n < N; n++) {
    let sum = 0;
    const rowOff = n * bytesPerRow;

    for (let b = 0; b < blocksPerRow; b++) {
      const bOff = rowOff + b * 18;
      // f16 scale — read little-endian uint16
      const scale = f16(raw[bOff] | (raw[bOff + 1] << 8));
      const dOff = bOff + 2;
      const xBase = b << 5;  // b * 32

      // Accumulate integer products, multiply by scale once per block
      let blockSum = 0;
      for (let j = 0; j < 16; j++) {
        const byte = raw[dOff + j];
        blockSum += x[xBase + j] * ((byte & 0x0F) - 8);
        blockSum += x[xBase + j + 16] * ((byte >>> 4) - 8);
      }
      sum += blockSum * scale;
    }

    output[n] = sum;
  }

  return output;
}

/**
 * Q4_1 matrix-vector multiply: y[N] = W_q4_1[N, K] @ x[K]
 *
 * @param {Float32Array} x — input vector [K]
 * @param {Buffer|Uint8Array} rawBuf — raw Q4_1 data, N rows × (K/32 × 20) bytes
 * @param {number} N — output dimension
 * @param {number} K — input dimension (must be multiple of 32)
 * @returns {Float32Array} — output [N]
 */
function q4_1_matvec(x, rawBuf, N, K) {
  const output = new Float32Array(N);
  const blocksPerRow = K >>> 5;
  const bytesPerRow = blocksPerRow * 20;
  const raw = rawBuf instanceof Uint8Array ? rawBuf :
    new Uint8Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength);

  for (let n = 0; n < N; n++) {
    let sum = 0;
    const rowOff = n * bytesPerRow;

    for (let b = 0; b < blocksPerRow; b++) {
      const bOff = rowOff + b * 20;
      const delta = f16(raw[bOff] | (raw[bOff + 1] << 8));
      const min = f16(raw[bOff + 2] | (raw[bOff + 3] << 8));
      const dOff = bOff + 4;
      const xBase = b << 5;

      // val = nibble * delta + min
      // sum += x[k] * (nibble * delta + min)
      //      = delta * sum(x[k] * nibble) + min * sum(x[k])
      let nibbleSum = 0;
      let xBlockSum = 0;
      for (let j = 0; j < 16; j++) {
        const byte = raw[dOff + j];
        nibbleSum += x[xBase + j] * (byte & 0x0F);
        nibbleSum += x[xBase + j + 16] * (byte >>> 4);
        xBlockSum += x[xBase + j] + x[xBase + j + 16];
      }
      sum += delta * nibbleSum + min * xBlockSum;
    }

    output[n] = sum;
  }

  return output;
}

/**
 * Dispatch quantized matrix-vector multiply based on type.
 * Uses WASM kernel when available (~5-8x faster than JS).
 */
let _wasmQ4Ready = false;
let _wasmQ4Disabled = false;
function initWasmQ4() {
  if (_wasmQ4Disabled) return false;
  if (!_wasmQ4Ready) {
    _wasmQ4Ready = wasmQ4.init();
    if (_wasmQ4Ready) console.log('  [WASM-Q4] Q4 matvec kernel ready');
  }
  return _wasmQ4Ready;
}

function quantizedMatvec(x, rawBuf, type, N, K) {
  if (_wasmQ4Ready || initWasmQ4()) {
    if (type === GGML_TYPE.Q4_0) return wasmQ4.q4_0_matvec(x, rawBuf, N, K);
    if (type === GGML_TYPE.Q4_1) return wasmQ4.q4_1_matvec(x, rawBuf, N, K);
  }
  // JS fallback
  if (type === GGML_TYPE.Q4_0) return q4_0_matvec(x, rawBuf, N, K);
  if (type === GGML_TYPE.Q4_1) return q4_1_matvec(x, rawBuf, N, K);
  throw new Error(`Unsupported quant type for matvec: ${type}`);
}

/** Disable/enable WASM Q4 kernels (for debugging). */
function setWasmQ4Disabled(v) { _wasmQ4Disabled = v; _wasmQ4Ready = false; }

class StreamingLlamaRuntime {
  /**
   * @param {Object} config — model configuration from GGUF header
   * @param {Object} options
   * @param {Object} options.loader — StreamingWeightLoader instance
   * @param {boolean} options.sharedWeights — whether lm_head = token_embedding
   * @param {boolean} options.noWasm — disable WASM SIMD
   * @param {number} options.earlyExitThreshold — confidence threshold for early exit (0 = disabled)
   * @param {number} options.earlyExitInterval — check every N layers (default 4)
   * @param {boolean} options.log — enable PureBee logging
   */
  constructor(config, options = {}) {
    this.config = config;
    this.gpu = new PureBee({ log: options.log || false });
    this.loader = options.loader;
    this._sharedWeights = options.sharedWeights !== false;
    this._kvInitialized = false;
    this._loaded = false;

    // Early exit settings
    this._earlyExitThreshold = options.earlyExitThreshold || 0;
    this._earlyExitInterval = options.earlyExitInterval || 4;
    this._lastExitLayer = -1;  // track which layer we exited at

    // WASM SIMD
    if (options.noWasm) {
      this._wasmReady = false;
    } else {
      this._wasmReady = this.gpu.engine.initWasm();
    }


    // Thread pool for parallel matvec
    this._threadPool = null;

    // Stats
    this._layersUsed = 0;
    this._totalForwards = 0;
    this._earlyExits = 0;
  }

  /**
   * Initialise worker thread pool for parallel matvec.
   * Call after loadResidentWeights(). Requires SharedArrayBuffer-backed raw cache.
   *
   * @param {number} [numWorkers] -- defaults to os.cpus().length - 1
   * @returns {Promise<boolean>} true if thread pool is ready
   */
  async initThreadPool(numWorkers) {
    const sharedBuffer = this.loader.getSharedBuffer ? this.loader.getSharedBuffer() : null;
    if (!sharedBuffer) {
      console.log('  [StreamingLLaMA] No SharedArrayBuffer -- skipping thread pool');
      return false;
    }

    const { dim, hiddenDim, vocabSize } = this.config;
    this._threadPool = new AtomicsThreadPool(numWorkers);
    const ok = await this._threadPool.init(sharedBuffer, { dim, hiddenDim, vocabSize });
    if (ok) {
      console.log(`  [StreamingLLaMA] Thread pool: ${this._threadPool.numThreads} threads (${this._threadPool.numThreads - 1} workers + main)`);
    }
    return ok;
  }

  /**
   * Load resident weights — embedding, final norm, lm_head.
   * These stay in memory for the entire session.
   */
  loadResidentWeights() {
    const gpu = this.gpu;
    const { dim, vocabSize } = this.config;

    console.log('  [StreamingLLaMA] Loading resident weights...');

    const resident = this.loader.loadResidentWeights(this._sharedWeights);

    // Token embedding [vocabSize, dim]
    gpu.GRID_WRITE('token_embedding', [vocabSize, dim], resident.tokenEmbedding);

    // Final RMS norm [dim]
    gpu.GRID_WRITE('rms_final', [dim], resident.rmsFinal);

    // LM head (if not shared)
    if (resident.lmHead) {
      gpu.GRID_WRITE('lm_head', [vocabSize, dim], resident.lmHead);
    }

    // Cache embedding in WASM if available and small enough
    // Large embeddings (128K vocab = ~1GB) would double memory usage in WASM copy
    const embSizeMB = (vocabSize * dim * 4) / (1024 * 1024);
    if (this._wasmReady && embSizeMB < 200) {
      const embTensor = gpu.GRID_READ('token_embedding');
      this.gpu.engine.cacheWeight(embTensor);
      this._embCached = true;
      const stats = this.gpu.engine.wasmStats;
      console.log(`  [StreamingLLaMA] WASM: cached embedding, ${stats.weightsMB}MB`);
    } else {
      this._embCached = false;
      if (this._wasmReady) {
        console.log(`  [StreamingLLaMA] WASM: embedding too large (${embSizeMB.toFixed(0)}MB), using JS for LM head`);
      }
    }

    this._loaded = true;
    const memStats = gpu.stats();
    console.log(`  [StreamingLLaMA] Resident: ${memStats.memory.tensors} tensors, ${memStats.memory.totalMB}MB`);

    // Initialize vocab pruner for large vocabularies
  }

  /**
   * Initialize KV cache for all layers.
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
   * Apply RoPE to Q and K tensors.
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

        for (let h = 0; h < nHeads; h++) {
          const idx = t * dim + h * headDim + i;
          const q0 = qData[idx];
          const q1 = qData[idx + 1];
          qData[idx]     = q0 * cosA - q1 * sinA;
          qData[idx + 1] = q0 * sinA + q1 * cosA;
        }

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
   * Run one transformer layer using streamed weights.
   * Loads weights from disk, computes, then discards.
   *
   * @param {number} l — layer index
   * @param {Object} layerWeights — { wq, wk, wv, wo, w1, w2, w3, rms_att, rms_ffn }
   * @param {number} seqLen
   * @param {number} startPos
   */
  _runLayer(l, layerWeights, seqLen, startPos) {
    const gpu = this.gpu;
    const { dim, hiddenDim, nHeads, nKvHeads, headDim } = this.config;
    const kvDim = nKvHeads * headDim;
    const cacheLen = startPos + seqLen;

    // Write layer weights into PureBee memory
    gpu.GRID_WRITE(`layer.rms_att`, [dim], layerWeights.rms_att);
    gpu.GRID_WRITE(`layer.rms_ffn`, [dim], layerWeights.rms_ffn);

    // Weight matrices — write and optionally cache in WASM
    const weightNames = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'];
    const weightShapes = {
      wq: [dim, dim], wk: [dim, kvDim], wv: [dim, kvDim], wo: [dim, dim],
      w1: [dim, hiddenDim], w2: [hiddenDim, dim], w3: [dim, hiddenDim],
    };

    // Reset WASM weight cache for this layer (reuse memory)
    if (this._wasmReady) {
      if (this._embCached) {
        // Embedding is cached — save pointer, reset, re-cache
        const embTensor = gpu.GRID_READ('token_embedding');
        const embPtr = embTensor._wasmPtr;
        wasmSimd.resetWeights();
        if (embPtr !== undefined) {
          embTensor._wasmPtr = wasmSimd.allocWeights(embTensor.data);
        }
      } else {
        // No embedding cached — just reset weights region
        wasmSimd.resetWeights();
      }
    }

    for (const name of weightNames) {
      const fullName = `layer.${name}`;
      gpu.GRID_WRITE(fullName, weightShapes[name], layerWeights[name]);

      // Cache in WASM for SIMD acceleration
      if (this._wasmReady) {
        const tensor = gpu.GRID_READ(fullName);
        this.gpu.engine.cacheWeight(tensor);
      }
    }

    // ── Pre-attention RMS Norm ──
    gpu.RMS_NORM('x', 'layer.rms_att', 'xnorm');

    // ── QKV Projections ──
    gpu.LINEAR('xnorm', 'layer.wq', null, 'q');
    gpu.LINEAR('xnorm', 'layer.wk', null, 'k');
    gpu.LINEAR('xnorm', 'layer.wv', null, 'v');

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
    const headsPerKvHead = nHeads / nKvHeads;
    const attnOut = new Float32Array(seqLen * dim);

    for (let t = 0; t < seqLen; t++) {
      for (let h = 0; h < nHeads; h++) {
        const kvH = Math.floor(h / headsPerKvHead);
        const scores = new Float32Array(cacheLen);
        const scale = 1.0 / Math.sqrt(headDim);

        for (let j = 0; j < cacheLen; j++) {
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

        let maxScore = -Infinity;
        for (let j = 0; j < cacheLen; j++) {
          if (scores[j] > maxScore) maxScore = scores[j];
        }
        let sumExp = 0;
        for (let j = 0; j < cacheLen; j++) {
          scores[j] = Math.exp(scores[j] - maxScore);
          sumExp += scores[j];
        }
        for (let j = 0; j < cacheLen; j++) scores[j] /= sumExp;

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
    gpu.LINEAR('attn_concat', 'layer.wo', null, 'attn_proj');

    // ── Residual ──
    gpu.TENSOR_ADD('x', 'attn_proj', 'x');

    // ── Pre-FFN RMS Norm ──
    gpu.RMS_NORM('x', 'layer.rms_ffn', 'xnorm');

    // ── SwiGLU FFN ──
    gpu.LINEAR('xnorm', 'layer.w1', null, 'ff_gate');
    gpu.SILU('ff_gate', 'ff_gate_act');
    gpu.LINEAR('xnorm', 'layer.w3', null, 'ff_up');
    gpu.ELEMENT_MUL('ff_gate_act', 'ff_up', 'ff_hidden');
    gpu.LINEAR('ff_hidden', 'layer.w2', null, 'ff_out');

    // ── Residual ──
    gpu.TENSOR_ADD('x', 'ff_out', 'x');

    // Free layer weights from PureBee memory
    for (const name of weightNames) {
      gpu.FREE(`layer.${name}`);
    }
    gpu.FREE('layer.rms_att');
    gpu.FREE('layer.rms_ffn');
  }

  /**
   * Run one transformer layer using raw Q4_0/Q4_1 weights directly.
   * Skips dequantisation, transposition, and WASM weight caching.
   * Only handles seqLen=1 (decode mode).
   * When thread pool is available, all matvecs are threaded across all cores.
   *
   * @param {number} l -- layer index
   * @param {Object} layerData -- from loader.loadLayerWeightsRaw()
   * @param {number} startPos
   */
  _runLayerQ4(l, layerData, startPos) {
    const gpu = this.gpu;
    const { dim, hiddenDim, nHeads, nKvHeads, headDim } = this.config;
    const kvDim = nKvHeads * headDim;
    const cacheLen = startPos + 1;
    const raw = layerData.raw;
    const tp = this._threadPool;
    const threaded = tp && tp.available;

    // Write norm weights (always F32, small)
    gpu.GRID_WRITE('layer.rms_att', [dim], layerData.rms_att);
    gpu.GRID_WRITE('layer.rms_ffn', [dim], layerData.rms_ffn);

    // ── Pre-attention RMS Norm ──
    gpu.RMS_NORM('x', 'layer.rms_att', 'xnorm');
    const xnormData = gpu.GRID_READ('xnorm').data;

    // ── QKV Projections (threaded when pool available) ──
    const mv = (x, w) => threaded
      ? tp.matvec(x, w.rawBuf, w.type, w.N, w.K)
      : quantizedMatvec(x, w.rawBuf, w.type, w.N, w.K);

    const qData = mv(xnormData, raw.wq);
    const kData = mv(xnormData, raw.wk);
    const vData = mv(xnormData, raw.wv);

    // ── RoPE ──
    this._applyRoPE(qData, kData, 1, startPos);

    // ── Update KV Cache ──
    const keyCache = gpu.GRID_READ(`kv_k_${l}`);
    const valCache = gpu.GRID_READ(`kv_v_${l}`);
    const dstOffset = startPos * kvDim;
    for (let d = 0; d < kvDim; d++) {
      keyCache.data[dstOffset + d] = kData[d];
      valCache.data[dstOffset + d] = vData[d];
    }

    // ── Multi-Head Attention ──
    const headsPerKvHead = nHeads / nKvHeads;
    const attnOut = new Float32Array(dim);
    const scale = 1.0 / Math.sqrt(headDim);

    for (let h = 0; h < nHeads; h++) {
      const kvH = Math.floor(h / headsPerKvHead);
      const scores = new Float32Array(cacheLen);

      for (let j = 0; j < cacheLen; j++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += qData[h * headDim + d] * keyCache.data[j * kvDim + kvH * headDim + d];
        }
        scores[j] = dot * scale;
      }

      let maxScore = -Infinity;
      for (let j = 0; j < cacheLen; j++) if (scores[j] > maxScore) maxScore = scores[j];
      let sumExp = 0;
      for (let j = 0; j < cacheLen; j++) { scores[j] = Math.exp(scores[j] - maxScore); sumExp += scores[j]; }
      for (let j = 0; j < cacheLen; j++) scores[j] /= sumExp;

      for (let d = 0; d < headDim; d++) {
        let val = 0;
        for (let j = 0; j < cacheLen; j++) {
          val += scores[j] * valCache.data[j * kvDim + kvH * headDim + d];
        }
        attnOut[h * headDim + d] = val;
      }
    }

    // ── Output Projection (threaded) ──
    const attnProj = mv(attnOut, raw.wo);

    // ── Residual ──
    const xData = gpu.GRID_READ('x');
    for (let d = 0; d < dim; d++) xData.data[d] += attnProj[d];

    // ── Pre-FFN RMS Norm ──
    gpu.RMS_NORM('x', 'layer.rms_ffn', 'xnorm');
    const xnormData2 = gpu.GRID_READ('xnorm').data;

    // ── SwiGLU FFN (threaded) ──
    const gateOut = mv(xnormData2, raw.w1);
    const upOut = mv(xnormData2, raw.w3);

    // SiLU(gate) * up
    const ffHidden = new Float32Array(hiddenDim);
    for (let i = 0; i < hiddenDim; i++) {
      const silu = gateOut[i] / (1 + Math.exp(-gateOut[i]));
      ffHidden[i] = silu * upOut[i];
    }

    // Down projection (threaded)
    const ffOut = mv(ffHidden, raw.w2);

    // ── Residual ──
    for (let d = 0; d < dim; d++) xData.data[d] += ffOut[d];

    // Free norm weights
    gpu.FREE('layer.rms_att');
    gpu.FREE('layer.rms_ffn');
  }


  /**
   * Compute logits from hidden state using LM head.
   * Expensive for large vocabularies (128K).
   *
   * @param {Float32Array} hiddenState — [dim]
   * @returns {Float32Array} — logits [vocabSize]
   */
  _computeLogits(hiddenState) {
    const { dim, vocabSize } = this.config;
    const gpu = this.gpu;

    // Apply final RMS norm
    gpu.GRID_WRITE('_exit_hidden', [1, dim], hiddenState);
    gpu.RMS_NORM('_exit_hidden', 'rms_final', '_exit_normed');
    const normed = gpu.GRID_READ('_exit_normed');

    const lmWeight = this._sharedWeights
      ? gpu.GRID_READ('token_embedding')
      : gpu.GRID_READ('lm_head');

    let logits;
    const tp = this._threadPool;
    const embIsShared = lmWeight.data.buffer instanceof SharedArrayBuffer;

    if (tp && tp.available && embIsShared) {
      // Threaded LM head: partition vocab rows across all cores
      logits = tp.lmHead(normed.data, lmWeight.data, vocabSize, dim);
    } else if (this._wasmReady && lmWeight._wasmPtr !== undefined) {
      // Embedding is WASM-cached -- use fast path (small vocabs)
      logits = new Float32Array(vocabSize);
      wasmSimd.matmulCached(normed.data, 1, dim, lmWeight._wasmPtr, vocabSize, null, logits);
    } else if (_wasmQ4Ready || initWasmQ4()) {
      // WASM SIMD batch_dot -- no transpose, process row-major chunks
      logits = wasmQ4.lmHead(normed.data, lmWeight.data, vocabSize, dim);
    } else if (this._wasmReady) {
      // Chunked WASM fallback (with transpose)
      logits = new Float32Array(vocabSize);
      const CHUNK = 256;
      const wData = lmWeight.data;
      const chunkBuf = new Float32Array(dim * CHUNK);
      const chunkOut = new Float32Array(CHUNK);

      for (let v0 = 0; v0 < vocabSize; v0 += CHUNK) {
        const chunkSize = Math.min(CHUNK, vocabSize - v0);
        for (let d = 0; d < dim; d++) {
          for (let c = 0; c < chunkSize; c++) {
            chunkBuf[d * chunkSize + c] = wData[(v0 + c) * dim + d];
          }
        }
        wasmSimd.matmul_general(normed.data, 1, dim, chunkBuf, chunkSize, chunkOut);
        logits.set(chunkOut.subarray(0, chunkSize), v0);
      }
    } else {
      // Pure JS fallback
      logits = new Float32Array(vocabSize);
      for (let v = 0; v < vocabSize; v++) {
        let dot = 0;
        const wOffset = v * dim;
        for (let d = 0; d < dim; d++) {
          dot += normed.data[d] * lmWeight.data[wOffset + d];
        }
        logits[v] = dot;
      }
    }

    // Cleanup temp tensors
    gpu.FREE('_exit_hidden');
    gpu.FREE('_exit_normed');

    return logits;
  }

  /**
   * Check if we should early-exit based on prediction confidence.
   *
   * @param {number} seqLen
   * @param {number} dim
   * @returns {{ shouldExit: boolean, topProb: number, logits: Float32Array|null }}
   */
  _checkEarlyExit(seqLen, dim) {
    const gpu = this.gpu;
    const xData = gpu.GRID_READ('x');

    // Extract last position's hidden state
    const lastOffset = (seqLen - 1) * dim;
    const hiddenState = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      hiddenState[d] = xData.data[lastOffset + d];
    }

    const logits = this._computeLogits(hiddenState);

    // Compute confidence from top logits
    // If pruning is active, only examine candidates (not all 128K)
    const topK = 8;
    const indexed = [];
    for (let i = 0; i < logits.length; i++) {
      indexed.push([logits[i], i]);
    }
    indexed.sort((a, b) => b[0] - a[0]);

    const maxLogit = indexed[0][0];
    let sumExp = 0;
    const topItems = indexed.slice(0, topK);
    for (let i = 0; i < topItems.length; i++) {
      sumExp += Math.exp(topItems[i][0] - maxLogit);
    }
    // Add approximate tail mass
    const tailCount = candidates ? candidates.length - topK : logits.length - topK;
    sumExp += Math.max(0, tailCount) * Math.exp(-20);

    const topProb = Math.exp(indexed[0][0] - maxLogit) / sumExp;

    return {
      shouldExit: topProb >= this._earlyExitThreshold,
      topProb,
      logits,
    };
  }

  /**
   * Forward pass through the streaming LLaMA transformer.
   *
   * @param {number[]} tokenIds — input tokens
   * @param {number} startPos — position in sequence (for KV cache)
   * @param {Object} options
   * @param {number} options.maxLayers — max layers to process (for draft mode)
   * @returns {{ logits: Float32Array, layersUsed: number }}
   */
  forward(tokenIds, startPos = 0, options = {}) {
    if (!this._loaded) throw new Error('Resident weights not loaded');
    if (!this._kvInitialized) this._initKVCache();

    const gpu = this.gpu;
    const { dim, nLayers, vocabSize } = this.config;
    const seqLen = tokenIds.length;
    const maxLayers = options.maxLayers || nLayers;
    const enableEarlyExit = this._earlyExitThreshold > 0 && seqLen === 1;

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

    // ── TRANSFORMER BLOCKS (streamed) ──
    let layersUsed = 0;
    let earlyExitLogits = null;

    const layerLimit = Math.min(maxLayers, nLayers);
    // Use Q4 fast path for single-token decode when raw cache is available
    const useQ4 = seqLen === 1 && this.loader.loadLayerWeightsRaw;

    for (let l = 0; l < layerLimit; l++) {
      if (useQ4) {
        // Q4 fast path: skip dequantization + transpose
        const layerData = this.loader.loadLayerWeightsRaw(l);
        this._runLayerQ4(l, layerData, startPos);
      } else {
        // Standard path (prefill or fallback)
        if (l + 1 < layerLimit) this.loader.prefetchLayer(l + 1);
        const layerWeights = this.loader.loadLayerWeights(l);
        this._runLayer(l, layerWeights, seqLen, startPos);
      }
      layersUsed++;

      // Early exit check (every N layers, skip the last layer)
      if (enableEarlyExit && l < nLayers - 1 && (l + 1) % this._earlyExitInterval === 0) {
        const exitCheck = this._checkEarlyExit(seqLen, dim);
        if (exitCheck.shouldExit) {
          this._earlyExits++;
          this._lastExitLayer = l;
          earlyExitLogits = exitCheck.logits;
          break;
        }
      }
    }

    this._layersUsed += layersUsed;
    this._totalForwards++;

    // ── FINAL LM HEAD ──
    let logits;
    if (earlyExitLogits) {
      logits = earlyExitLogits;
    } else {
      // Extract last position hidden state
      const xFinal = gpu.GRID_READ('x');
      const lastOffset = (seqLen - 1) * dim;
      const lastHidden = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        lastHidden[d] = xFinal.data[lastOffset + d];
      }
      logits = this._computeLogits(lastHidden);
      this._lastExitLayer = layersUsed - 1;
    }

    gpu.SYNC();
    return { logits, layersUsed };
  }

  /**
   * Async forward pass -- kept for API compatibility.
   * Threading is now synchronous via Atomics, so this just wraps forward().
   */
  async forwardAsync(tokenIds, startPos = 0, options = {}) {
    return this.forward(tokenIds, startPos, options);
  }

  /**
   * Reset KV cache.
   */
  resetCache() {
    if (this._kvInitialized) {
      const { nLayers, nKvHeads, headDim } = this.config;
      const kvDim = nKvHeads * headDim;
      for (let l = 0; l < nLayers; l++) {
        this.gpu.GRID_READ(`kv_k_${l}`).fill(0);
        this.gpu.GRID_READ(`kv_v_${l}`).fill(0);
      }
    }
  }

  /**
   * Rollback KV cache to a given position.
   * Used by speculative decoding when draft tokens are rejected.
   *
   * @param {number} pos — position to rollback to
   */
  rollbackCache(pos) {
    if (!this._kvInitialized) return;
    const { nLayers, nKvHeads, headDim, seqLen } = this.config;
    const kvDim = nKvHeads * headDim;

    for (let l = 0; l < nLayers; l++) {
      const keyCache = this.gpu.GRID_READ(`kv_k_${l}`);
      const valCache = this.gpu.GRID_READ(`kv_v_${l}`);
      // Zero out positions >= pos
      for (let p = pos; p < seqLen; p++) {
        const offset = p * kvDim;
        for (let d = 0; d < kvDim; d++) {
          keyCache.data[offset + d] = 0;
          valCache.data[offset + d] = 0;
        }
      }
    }
  }

  /**
   * Sample next token from logits.
   * Uses pruned candidate list when available to avoid 128K-element sort.
   */
  sample(logits, topK = 40, temperature = 0.8) {
    // Build indexed array — only from candidates if pruning is active
    let indexed;
    indexed = new Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      indexed[i] = [logits[i] / temperature, i];
    }
    indexed.sort((a, b) => b[0] - a[0]);
    const topKItems = indexed.slice(0, topK);

    const maxV = topKItems[0][0];
    let sum = 0;
    const probs = topKItems.map(([v]) => {
      const e = Math.exp(v - maxV);
      sum += e;
      return e;
    });
    const normalized = probs.map(p => p / sum);

    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < normalized.length; i++) {
      cumulative += normalized[i];
      if (r <= cumulative) return topKItems[i][1];
    }
    return topKItems[0][1];
  }

  /**
   * Greedy argmax.
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
   * Generate text autoregressively with layer streaming.
   */
  generate(promptTokens, maxTokens = 50, opts = {}) {
    const temperature = opts.temperature || 0.8;
    const topK = opts.topK || 40;
    const onToken = opts.onToken || null;
    const eosId = opts.eosId || 2;
    const greedy = opts.greedy || false;

    this.resetCache();
    this._layersUsed = 0;
    this._totalForwards = 0;
    this._earlyExits = 0;

    const allTokens = [...promptTokens];
    let pos = 0;

    // ── PREFILL ──
    const prefillStart = Date.now();
    const { logits } = this.forward(promptTokens, 0);
    const prefillTime = Date.now() - prefillStart;
    pos = promptTokens.length;

    let nextToken = greedy ? this.argmax(logits) : this.sample(logits, topK, temperature);
    allTokens.push(nextToken);
    if (onToken) onToken(nextToken, 0);

    // ── DECODE ──
    const decodeStart = Date.now();
    let generated = 1;

    for (let i = 1; i < maxTokens; i++) {
      if (nextToken === eosId) break;

      const { logits: stepLogits, layersUsed } = this.forward([nextToken], pos);
      pos++;

      nextToken = greedy ? this.argmax(stepLogits) : this.sample(stepLogits, topK, temperature);
      allTokens.push(nextToken);
      generated++;

      if (onToken) onToken(nextToken, i, layersUsed);
      if (nextToken === eosId) break;
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
      avgLayers: this._totalForwards > 0 ? (this._layersUsed / this._totalForwards).toFixed(1) : this.config.nLayers,
      earlyExits: this._earlyExits,
    };
  }

  /**
   * Async generate -- kept for API compatibility.
   * Threading is now synchronous via Atomics, so this just wraps generate().
   */
  async generateAsync(promptTokens, maxTokens = 50, opts = {}) {
    return this.generate(promptTokens, maxTokens, opts);
  }

  /**
   * Get runtime stats.
   */
  get stats() {
    return {
      totalForwards: this._totalForwards,
      avgLayers: this._totalForwards > 0 ? (this._layersUsed / this._totalForwards).toFixed(1) : 0,
      earlyExits: this._earlyExits,
      lastExitLayer: this._lastExitLayer,
      wasmReady: this._wasmReady,
    };
  }

  async shutdown() {
    if (this._threadPool) {
      await this._threadPool.shutdown();
      this._threadPool = null;
    }
    if (this.loader) {
      this.loader.close();
    }
  }
}

module.exports = { StreamingLlamaRuntime, setWasmQ4Disabled };
