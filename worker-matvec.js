/**
 * PureBee -- Worker Thread for Parallel Matvec (Atomics-based)
 *
 * Spins on Atomics.wait, reads work descriptors from shared control buffer,
 * computes Q4 matvec or LM head dot products on assigned row ranges,
 * writes results directly to shared output buffer.
 *
 * Shared memory layout (per worker slot, 64-byte aligned):
 *   Int32 [0]: status  -- 0=IDLE, 1=WORK_READY, 255=SHUTDOWN
 *   Int32 [1]: opType  -- 1=Q4_MATVEC, 2=LM_HEAD
 *   Int32 [2]: weightOffset (byte offset into weightBuf)
 *   Int32 [3]: weightLength (bytes)
 *   Int32 [4]: startRow
 *   Int32 [5]: endRow
 *   Int32 [6]: N (total rows)
 *   Int32 [7]: K (input dimension)
 *   Int32 [8]: quantType (2=Q4_0, 3=Q4_1)
 *   Int32 [9]: embOffset (byte offset for LM head embedding data)
 *
 * ioBuf layout:
 *   [0, maxK*4):                input vector x (main writes, workers read)
 *   [maxK*4, maxK*4+maxN*4):   output vector y (each worker writes its row range)
 *
 * Zero external dependencies.
 */

'use strict';

const { workerData } = require('worker_threads');
const wasmQ4 = require('./wasm-q4');

const { controlBuf, weightBuf, ioBuf, workerId, maxK } = workerData;

// Status constants
const IDLE = 0;
const WORK_READY = 1;
const SHUTDOWN = 255;

// Op types
const OP_Q4_MATVEC = 1;
const OP_LM_HEAD = 2;

// Shared typed arrays
const control = new Int32Array(controlBuf);
// control is Int32Array over the control SharedArrayBuffer

// Worker slot offset in control buffer (Int32 indices)
// Header: 16 Int32s (64 bytes). Each slot: 16 Int32s (64 bytes).
const HEADER_INTS = 16;
const SLOT_INTS = 16;
const slotBase = HEADER_INTS + workerId * SLOT_INTS;

// Slot field offsets (relative to slotBase)
const F_STATUS = 0;
const F_OP_TYPE = 1;
const F_WEIGHT_OFFSET = 2;
const F_START_ROW = 4;
const F_END_ROW = 5;
const F_K = 7;
const F_QUANT_TYPE = 8;
const F_EMB_OFFSET = 9;

// Initialise own WASM Q4 instance
const wasmOk = wasmQ4.init();
if (!wasmOk) {
  // Signal failure by not incrementing ready count
  process.exit(1);
}

// ioBuf regions
const inputByteOffset = 0;
const outputByteOffset = maxK * 4;

// Signal ready: increment readyCount (header Int32[1])
Atomics.add(control, 1, 1);
Atomics.notify(control, 1);

// Main work loop
while (true) {
  // Wait for work (status transitions from IDLE to WORK_READY or SHUTDOWN)
  const statusIdx = slotBase + F_STATUS;
  Atomics.wait(control, statusIdx, IDLE);

  const status = Atomics.load(control, statusIdx);
  if (status === SHUTDOWN) break;
  if (status !== WORK_READY) {
    // Spurious wakeup, go back to waiting
    continue;
  }

  // Read work descriptor
  const opType = control[slotBase + F_OP_TYPE];
  const weightOffset = control[slotBase + F_WEIGHT_OFFSET];
  const startRow = control[slotBase + F_START_ROW];
  const endRow = control[slotBase + F_END_ROW];
  const K = control[slotBase + F_K];
  const quantType = control[slotBase + F_QUANT_TYPE];
  const embOffset = control[slotBase + F_EMB_OFFSET];

  const sliceN = endRow - startRow;

  if (opType === OP_Q4_MATVEC) {
    // Read input vector x from ioBuf
    const x = new Float32Array(ioBuf, inputByteOffset, K);

    // Compute byte offset for our row slice within the weight data
    const blocksPer = K >>> 5;
    const bytesPerRow = quantType === 2 ? blocksPer * 18 : blocksPer * 20;
    const sliceByteOffset = weightOffset + startRow * bytesPerRow;
    const sliceByteLength = sliceN * bytesPerRow;
    const rawSlice = new Uint8Array(weightBuf, sliceByteOffset, sliceByteLength);

    // Run WASM SIMD kernel
    let output;
    if (quantType === 2) {
      output = wasmQ4.q4_0_matvec(x, rawSlice, sliceN, K);
    } else {
      output = wasmQ4.q4_1_matvec(x, rawSlice, sliceN, K);
    }

    // Write output directly to ioBuf at the correct offset
    const outView = new Float32Array(ioBuf, outputByteOffset + startRow * 4, sliceN);
    outView.set(output);

  } else if (opType === OP_LM_HEAD) {
    // Read hidden vector from ioBuf input region
    const hidden = new Float32Array(ioBuf, inputByteOffset, K);

    // Embedding data slice from weightBuf
    const dim = K;
    const embStart = embOffset + startRow * dim * 4;
    const embSlice = new Float32Array(weightBuf, embStart, sliceN * dim);

    // Compute logits for our row slice
    const logits = wasmQ4.lmHead(hidden, embSlice, sliceN, dim);

    // Write to output region
    const outView = new Float32Array(ioBuf, outputByteOffset + startRow * 4, sliceN);
    outView.set(logits);
  }

  // Signal done: set status back to IDLE and notify main thread
  Atomics.store(control, statusIdx, IDLE);
  Atomics.notify(control, statusIdx);
}
