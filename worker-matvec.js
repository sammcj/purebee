/**
 * PureBee — Worker Thread for Parallel Matvec
 *
 * Runs WASM SIMD Q4 matvec kernels on a slice of output rows.
 * Receives raw weight data via SharedArrayBuffer (zero-copy).
 * Input vector x is copied via postMessage (small — dim * 4 bytes).
 *
 * Protocol:
 *   init:   { type: 'init', sharedBuffer: SharedArrayBuffer }
 *   matvec: { type: 'matvec', id, xBuf, rawOffset, rawLength, quantType, startRow, endRow, N, K }
 *   lmhead: { type: 'lmhead', id, hiddenBuf, embOffset, embLength, startRow, endRow, vocabSize, dim }
 *   result: { type: 'result', id, output: Float32Array }
 */

'use strict';

const { parentPort } = require('worker_threads');
const wasmQ4 = require('./wasm-q4');

let sharedBuffer = null;
let ready = false;

// Initialize WASM Q4 engine
const wasmOk = wasmQ4.init();
if (!wasmOk) {
  parentPort.postMessage({ type: 'error', msg: 'WASM Q4 init failed in worker' });
}

parentPort.on('message', (msg) => {
  switch (msg.type) {
    case 'init': {
      sharedBuffer = msg.sharedBuffer;
      ready = true;
      parentPort.postMessage({ type: 'ready' });
      break;
    }

    case 'matvec': {
      if (!ready) {
        parentPort.postMessage({ type: 'error', id: msg.id, msg: 'worker not initialized' });
        return;
      }

      const { id, quantType, startRow, endRow, N, K } = msg;
      const sliceN = endRow - startRow;

      // x is transferred (ownership moved to worker)
      const x = new Float32Array(msg.xBuf);

      // Raw weight data: slice from SharedArrayBuffer
      // Each row's raw data is contiguous. Compute byte offset for the row slice.
      const blocksPer = K >>> 5;
      const bytesPerRow = quantType === 2 ? blocksPer * 18 : blocksPer * 20; // Q4_0=18, Q4_1=20
      const sliceByteOffset = msg.rawOffset + startRow * bytesPerRow;
      const sliceByteLength = sliceN * bytesPerRow;
      const rawSlice = new Uint8Array(sharedBuffer, sliceByteOffset, sliceByteLength);

      // Run WASM SIMD kernel on the row slice
      let output;
      if (quantType === 2) {
        output = wasmQ4.q4_0_matvec(x, rawSlice, sliceN, K);
      } else {
        output = wasmQ4.q4_1_matvec(x, rawSlice, sliceN, K);
      }

      // Transfer the output buffer back (zero-copy)
      parentPort.postMessage(
        { type: 'result', id, output: output.buffer, startRow, endRow },
        [output.buffer]
      );
      break;
    }

    case 'lmhead': {
      if (!ready) {
        parentPort.postMessage({ type: 'error', id: msg.id, msg: 'worker not initialized' });
        return;
      }

      const { id, startRow, endRow, vocabSize, dim } = msg;
      const hidden = new Float32Array(msg.hiddenBuf);
      const sliceN = endRow - startRow;

      // Embedding data from shared buffer
      const embStart = msg.embOffset + startRow * dim * 4;
      const embSlice = new Float32Array(sharedBuffer, embStart, sliceN * dim);

      // Use lmHead with the slice
      const logits = wasmQ4.lmHead(hidden, embSlice, sliceN, dim);

      parentPort.postMessage(
        { type: 'result', id, output: logits.buffer, startRow, endRow },
        [logits.buffer]
      );
      break;
    }
  }
});
