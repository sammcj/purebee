/**
 * PureBee — Thread Pool for Parallel Matvec
 *
 * Creates a single worker thread at startup. Splits large matvec
 * operations across main thread and worker for ~2x speedup.
 *
 * Design:
 *   - Worker receives SharedArrayBuffer reference once at init
 *   - For each matvec: worker computes top half, main thread computes bottom half
 *   - Input vector x is small (dim * 4 bytes) — copied via postMessage
 *   - Output is transferred back (zero-copy via Transferable)
 *   - Small matrices (N < threshold) run on main thread only
 *
 * Zero external dependencies.
 */

'use strict';

const { Worker } = require('worker_threads');
const path = require('path');
const wasmQ4 = require('./wasm-q4');

// Minimum output rows to justify threading overhead (~1-2ms message cost)
// Only thread large FFN matrices (8192 rows), not attention (2048 rows)
const MIN_ROWS_FOR_THREADING = 4096;

class MatvecThreadPool {
  constructor() {
    this._worker = null;
    this._ready = false;
    this._pendingInit = null;
    this._pending = new Map();  // id → { resolve, reject }
    this._nextId = 0;
    this._sharedBuffer = null;
  }

  /**
   * Initialize the thread pool with one worker.
   * @param {SharedArrayBuffer} sharedBuffer — raw weight cache
   * @returns {Promise<boolean>}
   */
  async init(sharedBuffer) {
    if (!sharedBuffer || !(sharedBuffer instanceof SharedArrayBuffer)) {
      console.log('  [ThreadPool] No SharedArrayBuffer — single-threaded mode');
      return false;
    }

    this._sharedBuffer = sharedBuffer;

    return new Promise((resolve) => {
      const workerPath = path.join(__dirname, 'worker-matvec.js');
      this._worker = new Worker(workerPath);

      this._worker.on('message', (msg) => {
        if (msg.type === 'ready') {
          this._ready = true;
          console.log('  [ThreadPool] Worker ready');
          resolve(true);
        } else if (msg.type === 'result') {
          const p = this._pending.get(msg.id);
          if (p) {
            this._pending.delete(msg.id);
            p.resolve(msg);
          }
        } else if (msg.type === 'error') {
          console.warn('  [ThreadPool] Worker error:', msg.msg);
          if (msg.id !== undefined) {
            const p = this._pending.get(msg.id);
            if (p) {
              this._pending.delete(msg.id);
              p.reject(new Error(msg.msg));
            }
          }
          if (!this._ready) resolve(false);
        }
      });

      this._worker.on('error', (err) => {
        console.warn('  [ThreadPool] Worker crashed:', err.message);
        this._ready = false;
        if (!this._ready) resolve(false);
      });

      // Send shared buffer to worker
      this._worker.postMessage({ type: 'init', sharedBuffer });
    });
  }

  /**
   * Check if thread pool is available.
   */
  get available() {
    return this._ready;
  }

  /**
   * Dispatch a matvec to the worker thread. Returns a promise.
   * @private
   */
  _dispatchMatvec(xBuf, rawOffset, rawLength, quantType, startRow, endRow, N, K) {
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage(
        { type: 'matvec', id, xBuf, rawOffset, rawLength, quantType, startRow, endRow, N, K },
        [xBuf]  // transfer x buffer to worker
      );
    });
  }

  /**
   * Dispatch an LM head slice to the worker. Returns a promise.
   * @private
   */
  _dispatchLmHead(hiddenBuf, embOffset, embLength, startRow, endRow, vocabSize, dim) {
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage(
        { type: 'lmhead', id, hiddenBuf, embOffset, embLength, startRow, endRow, vocabSize, dim },
        [hiddenBuf]
      );
    });
  }

  /**
   * Split a Q4 matvec across main thread and worker.
   * Worker gets the top half of rows, main thread gets the bottom half.
   * Both run simultaneously; results are concatenated.
   *
   * @param {Float32Array} x — input vector [K]
   * @param {Uint8Array} rawBuf — raw Q4 weight data (view into SharedArrayBuffer)
   * @param {number} quantType — GGML_TYPE.Q4_0 (2) or Q4_1 (3)
   * @param {number} N — total output rows
   * @param {number} K — input dimension
   * @returns {Promise<Float32Array>} — output vector [N]
   */
  async splitMatvec(x, rawBuf, quantType, N, K) {
    // Skip threading for small matrices
    if (!this._ready || N < MIN_ROWS_FOR_THREADING) {
      if (quantType === 2) return wasmQ4.q4_0_matvec(x, rawBuf, N, K);
      return wasmQ4.q4_1_matvec(x, rawBuf, N, K);
    }

    // Split rows: worker gets top half, main thread gets bottom half
    const splitRow = N >>> 1;

    // Copy x for the worker (x is small — dim*4 bytes)
    const xCopy = new Float32Array(K);
    xCopy.set(x);

    // rawBuf must be a view into the SharedArrayBuffer
    const rawOffset = rawBuf.byteOffset;
    const rawLength = rawBuf.byteLength;

    // Dispatch top half to worker
    const workerPromise = this._dispatchMatvec(
      xCopy.buffer, rawOffset, rawLength, quantType,
      0, splitRow, N, K
    );

    // Compute bottom half on main thread
    const blocksPer = K >>> 5;
    const bytesPerRow = quantType === 2 ? blocksPer * 18 : blocksPer * 20;
    const mainOffset = splitRow * bytesPerRow;
    const mainN = N - splitRow;
    const mainRaw = new Uint8Array(rawBuf.buffer, rawBuf.byteOffset + mainOffset, mainN * bytesPerRow);

    let mainResult;
    if (quantType === 2) {
      mainResult = wasmQ4.q4_0_matvec(x, mainRaw, mainN, K);
    } else {
      mainResult = wasmQ4.q4_1_matvec(x, mainRaw, mainN, K);
    }

    // Wait for worker result
    const workerMsg = await workerPromise;
    const workerResult = new Float32Array(workerMsg.output);

    // Concatenate: [worker top half | main bottom half]
    const output = new Float32Array(N);
    output.set(workerResult, 0);
    output.set(mainResult, splitRow);

    return output;
  }

  /**
   * Split LM head computation across main thread and worker.
   *
   * @param {Float32Array} hidden — normalized hidden state [dim]
   * @param {Float32Array} embData — embedding weights [vocabSize * dim] (in SharedArrayBuffer)
   * @param {number} vocabSize — total vocab size
   * @param {number} dim — embedding dimension
   * @returns {Promise<Float32Array>} — logits [vocabSize]
   */
  async splitLmHead(hidden, embData, vocabSize, dim) {
    if (!this._ready) {
      return wasmQ4.lmHead(hidden, embData, vocabSize, dim);
    }

    const splitRow = vocabSize >>> 1;

    // Copy hidden for the worker
    const hiddenCopy = new Float32Array(dim);
    hiddenCopy.set(hidden);

    // embData must be backed by SharedArrayBuffer
    const embOffset = embData.byteOffset;
    const embLength = embData.byteLength;

    // Dispatch top half to worker
    const workerPromise = this._dispatchLmHead(
      hiddenCopy.buffer, embOffset, embLength,
      0, splitRow, vocabSize, dim
    );

    // Compute bottom half on main thread
    const mainEmb = new Float32Array(embData.buffer, embData.byteOffset + splitRow * dim * 4, (vocabSize - splitRow) * dim);
    const mainResult = wasmQ4.lmHead(hidden, mainEmb, vocabSize - splitRow, dim);

    // Wait for worker
    const workerMsg = await workerPromise;
    const workerResult = new Float32Array(workerMsg.output);

    // Concatenate
    const logits = new Float32Array(vocabSize);
    logits.set(workerResult, 0);
    logits.set(mainResult, splitRow);

    return logits;
  }

  /**
   * Shutdown the worker thread.
   */
  async shutdown() {
    if (this._worker) {
      await this._worker.terminate();
      this._worker = null;
      this._ready = false;
    }
  }
}

module.exports = { MatvecThreadPool };
