/**
 * PureBee -- Atomics-Based Multi-Core Thread Pool
 *
 * Spawns N worker threads that spin on Atomics.wait. Dispatches matvec
 * work via shared memory descriptors + Atomics.notify. Main thread
 * computes its own partition simultaneously. Fully synchronous from
 * the caller's perspective.
 *
 * Shared memory layout:
 *   controlBuf: 64-byte header + 64 bytes per worker slot
 *     Header Int32[0]: numWorkers
 *     Header Int32[1]: readyCount (workers increment on init)
 *     Slot Int32[0]: status (0=IDLE, 1=WORK_READY, 255=SHUTDOWN)
 *     Slot Int32[1]: opType (1=Q4_MATVEC, 2=LM_HEAD)
 *     Slot Int32[2..9]: work descriptor fields
 *
 *   ioBuf: shared input/output vectors
 *     [0, maxK*4): input vector x
 *     [maxK*4, maxK*4+maxN*4): output vector y
 *
 * Zero external dependencies.
 */

'use strict';

const { Worker } = require('worker_threads');
const os = require('os');
const path = require('path');
const wasmQ4 = require('./wasm-q4');

// Status constants (must match worker-matvec.js)
const WORK_READY = 1;
const SHUTDOWN = 255;

// Op types
const OP_Q4_MATVEC = 1;
const OP_LM_HEAD = 2;

// Control buffer layout
const HEADER_INTS = 16;  // 64 bytes
const SLOT_INTS = 16;    // 64 bytes per worker

// Slot field offsets
const F_STATUS = 0;
const F_OP_TYPE = 1;
const F_WEIGHT_OFFSET = 2;
const F_START_ROW = 4;
const F_END_ROW = 5;
const F_N = 6;
const F_K = 7;
const F_QUANT_TYPE = 8;
const F_EMB_OFFSET = 9;

// Atomics.wait timeout (safety net)
const WAIT_TIMEOUT_MS = 5000;

class AtomicsThreadPool {
  /**
   * @param {number} [numWorkers] -- defaults to os.cpus().length - 1
   */
  constructor(numWorkers) {
    this._numWorkers = numWorkers || Math.max(1, os.cpus().length - 1);
    this._workers = [];
    this._ready = false;
    this._controlBuf = null;
    this._control = null;
    this._ioBuf = null;
    this._weightBuf = null;
    this._maxK = 0;
    this._maxN = 0;
  }

  /**
   * Spawn workers and allocate shared buffers.
   *
   * @param {SharedArrayBuffer} weightBuf -- raw weight cache from streaming-loader
   * @param {Object} config -- { dim, hiddenDim, vocabSize }
   * @returns {Promise<boolean>}
   */
  async init(weightBuf, config) {
    if (!weightBuf || !(weightBuf instanceof SharedArrayBuffer)) {
      console.log('  [ThreadPool] No SharedArrayBuffer -- single-threaded mode');
      return false;
    }

    this._weightBuf = weightBuf;

    // Compute max dimensions for ioBuf sizing
    // maxK: largest input dimension (hiddenDim for w2 down-projection)
    // maxN: largest output dimension (vocabSize for LM head)
    this._maxK = Math.max(config.dim, config.hiddenDim);
    this._maxN = Math.max(config.vocabSize, config.hiddenDim);

    // Allocate control buffer: header + slots for all workers
    const controlBytes = (HEADER_INTS + this._numWorkers * SLOT_INTS) * 4;
    this._controlBuf = new SharedArrayBuffer(controlBytes);
    this._control = new Int32Array(this._controlBuf);
    this._control[0] = this._numWorkers;
    this._control[1] = 0; // readyCount

    // Allocate I/O buffer: input x + output y
    const ioBytes = (this._maxK + this._maxN) * 4;
    this._ioBuf = new SharedArrayBuffer(ioBytes);

    // Spawn workers
    const workerPath = path.join(__dirname, 'worker-matvec.js');

    for (let i = 0; i < this._numWorkers; i++) {
      const worker = new Worker(workerPath, {
        workerData: {
          controlBuf: this._controlBuf,
          weightBuf: this._weightBuf,
          ioBuf: this._ioBuf,
          workerId: i,
          maxK: this._maxK,
          maxN: this._maxN,
        },
      });

      worker.on('error', (err) => {
        console.warn(`  [ThreadPool] Worker ${i} error:`, err.message);
      });

      this._workers.push(worker);
    }

    // Wait for all workers to signal ready
    const deadline = Date.now() + WAIT_TIMEOUT_MS;
    while (Atomics.load(this._control, 1) < this._numWorkers) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) {
        console.warn(`  [ThreadPool] Timeout waiting for workers (${Atomics.load(this._control, 1)}/${this._numWorkers} ready)`);
        await this.shutdown();
        return false;
      }
      Atomics.wait(this._control, 1, Atomics.load(this._control, 1), Math.min(remaining, 100));
    }

    this._ready = true;
    console.log(`  [ThreadPool] ${this._numWorkers} workers ready`);
    return true;
  }

  get available() {
    return this._ready;
  }

  get numThreads() {
    return this._ready ? this._numWorkers + 1 : 1;
  }

  /**
   * Synchronous threaded Q4 matvec: y[N] = W_q4[N,K] @ x[K]
   *
   * Partitions rows across (numWorkers + 1) threads. Main thread
   * computes its own partition while workers run in parallel.
   *
   * @param {Float32Array} x -- input vector [K]
   * @param {Uint8Array} rawBuf -- raw Q4 weight data (view into weightBuf)
   * @param {number} quantType -- 2=Q4_0, 3=Q4_1
   * @param {number} N -- output rows
   * @param {number} K -- input dimension
   * @returns {Float32Array} -- output [N]
   */
  matvec(x, rawBuf, quantType, N, K) {
    const totalThreads = this._numWorkers + 1;
    const activeThreads = Math.min(totalThreads, N);
    const activeWorkers = activeThreads - 1; // reserve 1 for main

    // Write input x into ioBuf
    const ioX = new Float32Array(this._ioBuf, 0, K);
    ioX.set(x);

    // Partition rows evenly
    const rowsPerThread = Math.floor(N / activeThreads);
    const extraRows = N % activeThreads;

    // Weight data byte offset (rawBuf is a view into weightBuf)
    const weightOffset = rawBuf.byteOffset;
    const blocksPer = K >>> 5;
    const bytesPerRow = quantType === 2 ? blocksPer * 18 : blocksPer * 20;

    // Dispatch to workers
    let rowStart = 0;
    for (let w = 0; w < activeWorkers; w++) {
      const rows = rowsPerThread + (w < extraRows ? 1 : 0);
      const rowEnd = rowStart + rows;

      const slotBase = HEADER_INTS + w * SLOT_INTS;
      this._control[slotBase + F_OP_TYPE] = OP_Q4_MATVEC;
      this._control[slotBase + F_WEIGHT_OFFSET] = weightOffset;
      this._control[slotBase + F_START_ROW] = rowStart;
      this._control[slotBase + F_END_ROW] = rowEnd;
      this._control[slotBase + F_N] = N;
      this._control[slotBase + F_K] = K;
      this._control[slotBase + F_QUANT_TYPE] = quantType;

      // Signal worker
      Atomics.store(this._control, slotBase + F_STATUS, WORK_READY);
      Atomics.notify(this._control, slotBase + F_STATUS);

      rowStart = rowEnd;
    }

    // Main thread computes remaining rows
    const mainStartRow = rowStart;
    const mainN = N - mainStartRow;

    if (mainN > 0) {
      const mainByteOffset = mainStartRow * bytesPerRow;
      const mainRaw = new Uint8Array(
        rawBuf.buffer, rawBuf.byteOffset + mainByteOffset, mainN * bytesPerRow
      );

      let mainResult;
      if (quantType === 2) {
        mainResult = wasmQ4.q4_0_matvec(x, mainRaw, mainN, K);
      } else {
        mainResult = wasmQ4.q4_1_matvec(x, mainRaw, mainN, K);
      }

      // Write main thread's result to ioBuf
      const outView = new Float32Array(
        this._ioBuf, this._maxK * 4 + mainStartRow * 4, mainN
      );
      outView.set(mainResult);
    }

    // Wait for all workers to finish
    for (let w = 0; w < activeWorkers; w++) {
      const statusIdx = HEADER_INTS + w * SLOT_INTS + F_STATUS;
      let waitResult = Atomics.wait(this._control, statusIdx, WORK_READY, WAIT_TIMEOUT_MS);
      if (waitResult === 'timed-out') {
        // Worker still running -- wait once more with extended timeout
        waitResult = Atomics.wait(this._control, statusIdx, WORK_READY, WAIT_TIMEOUT_MS);
        if (waitResult === 'timed-out') {
          throw new Error(`ThreadPool: worker ${w} timed out on matvec (10s)`);
        }
      }
    }

    // Read combined output from ioBuf
    const output = new Float32Array(N);
    output.set(new Float32Array(this._ioBuf, this._maxK * 4, N));
    return output;
  }

  /**
   * Synchronous threaded LM head: logits[vocabSize] = embedding[vocabSize,dim] @ hidden[dim]
   *
   * @param {Float32Array} hidden -- normalised hidden state [dim]
   * @param {Float32Array} embData -- embedding weights (in SharedArrayBuffer)
   * @param {number} vocabSize -- total vocab size
   * @param {number} dim -- embedding dimension
   * @returns {Float32Array} -- logits [vocabSize]
   */
  lmHead(hidden, embData, vocabSize, dim) {
    const totalThreads = this._numWorkers + 1;
    const activeThreads = Math.min(totalThreads, vocabSize);
    const activeWorkers = activeThreads - 1;

    // Write hidden into ioBuf input region
    const ioX = new Float32Array(this._ioBuf, 0, dim);
    ioX.set(hidden);

    const rowsPerThread = Math.floor(vocabSize / activeThreads);
    const extraRows = vocabSize % activeThreads;

    // embData byte offset within its backing SharedArrayBuffer
    const embOffset = embData.byteOffset;

    // Dispatch to workers
    let rowStart = 0;
    for (let w = 0; w < activeWorkers; w++) {
      const rows = rowsPerThread + (w < extraRows ? 1 : 0);
      const rowEnd = rowStart + rows;

      const slotBase = HEADER_INTS + w * SLOT_INTS;
      this._control[slotBase + F_OP_TYPE] = OP_LM_HEAD;
      this._control[slotBase + F_EMB_OFFSET] = embOffset;
      this._control[slotBase + F_START_ROW] = rowStart;
      this._control[slotBase + F_END_ROW] = rowEnd;
      this._control[slotBase + F_K] = dim;

      Atomics.store(this._control, slotBase + F_STATUS, WORK_READY);
      Atomics.notify(this._control, slotBase + F_STATUS);

      rowStart = rowEnd;
    }

    // Main thread computes remaining rows
    const mainStartRow = rowStart;
    const mainN = vocabSize - mainStartRow;

    if (mainN > 0) {
      const mainEmb = new Float32Array(
        embData.buffer, embData.byteOffset + mainStartRow * dim * 4, mainN * dim
      );
      const mainResult = wasmQ4.lmHead(hidden, mainEmb, mainN, dim);

      const outView = new Float32Array(
        this._ioBuf, this._maxK * 4 + mainStartRow * 4, mainN
      );
      outView.set(mainResult);
    }

    // Wait for workers
    for (let w = 0; w < activeWorkers; w++) {
      const statusIdx = HEADER_INTS + w * SLOT_INTS + F_STATUS;
      let waitResult = Atomics.wait(this._control, statusIdx, WORK_READY, WAIT_TIMEOUT_MS);
      if (waitResult === 'timed-out') {
        waitResult = Atomics.wait(this._control, statusIdx, WORK_READY, WAIT_TIMEOUT_MS);
        if (waitResult === 'timed-out') {
          throw new Error(`ThreadPool: worker ${w} timed out on lmHead (10s)`);
        }
      }
    }

    // Read combined output
    const logits = new Float32Array(vocabSize);
    logits.set(new Float32Array(this._ioBuf, this._maxK * 4, vocabSize));
    return logits;
  }

  /**
   * Shutdown all worker threads.
   */
  async shutdown() {
    if (!this._control) return;

    for (let w = 0; w < this._workers.length; w++) {
      const slotBase = HEADER_INTS + w * SLOT_INTS;
      Atomics.store(this._control, slotBase + F_STATUS, SHUTDOWN);
      Atomics.notify(this._control, slotBase + F_STATUS);
    }

    const terminations = this._workers.map(w => w.terminate());
    await Promise.all(terminations);

    this._workers = [];
    this._ready = false;
  }
}

module.exports = { AtomicsThreadPool };
