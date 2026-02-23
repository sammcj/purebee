/**
 * PureBee — 4 — WASM SIMD Matmul Kernel
 *
 * Hardware-accelerated matrix multiplication using WebAssembly SIMD.
 * The WASM binary is built programmatically — zero external dependencies.
 *
 * Key features:
 *   - f32x4 SIMD: process 4 floats per cycle in the inner loop
 *   - Weight caching: weights are copied to WASM memory once during loading,
 *     eliminating per-matmul copy overhead
 *   - Scratch region: small activation/output buffers reused each call
 *
 * Memory layout:
 *   [0, SCRATCH_SIZE)          — scratch for activations + outputs
 *   [SCRATCH_SIZE, ...)        — persistent weight storage (bump-allocated)
 *
 * Zero external dependencies.
 */

'use strict';

// ─── Configuration ────────────────────────────────────────────────

const SCRATCH_SIZE = 4 * 1024 * 1024;  // 4MB scratch region
const INITIAL_PAGES = 256;              // 16MB initial WASM memory

// ─── Module State ─────────────────────────────────────────────────

let wasmInstance = null;
let wasmMemory = null;
let memoryPages = 0;
let weightBumpPtr = SCRATCH_SIZE;
let _supported = false;

// ─── WASM Binary Encoding Helpers ─────────────────────────────────

function uleb128(n) {
  const r = [];
  do {
    let b = n & 0x7f;
    n >>>= 7;
    if (n) b |= 0x80;
    r.push(b);
  } while (n);
  return r;
}

function sleb128(n) {
  const r = [];
  let more = true;
  while (more) {
    let b = n & 0x7f;
    n >>= 7;
    if ((n === 0 && !(b & 0x40)) || (n === -1 && (b & 0x40))) more = false;
    else b |= 0x80;
    r.push(b);
  }
  return r;
}

function encStr(s) {
  return [...uleb128(s.length), ...Array.from(s, c => c.charCodeAt(0))];
}

function section(id, data) {
  return [id, ...uleb128(data.length), ...data];
}

// ─── Build WASM Module ────────────────────────────────────────────
//
// Equivalent WAT:
//
//   (module
//     (memory (import "env" "memory") 1)
//     (func (export "matmul")
//       (param $aPtr i32) (param $bPtr i32) (param $cPtr i32)
//       (param $M i32) (param $K i32) (param $N i32)
//       (local $m i32) (local $k i32) (local $n i32)
//       (local $aOff i32) (local $bOff i32) (local $cOff i32)
//       (local $N4 i32) (local $aVal f32) (local $aVec v128)
//
//       ;; N4 = N & ~3    (round down to multiple of 4)
//       ;; for m = 0..M:
//       ;;   for k = 0..K:
//       ;;     aVal = A[m*K+k]; aVec = f32x4.splat(aVal)
//       ;;     bOff = &B[k,0]; cOff = &C[m,0]
//       ;;     for n = 0..N4 step 4:   (SIMD)
//       ;;       C[m,n:n+4] += aVec * B[k,n:n+4]
//       ;;     for n = N4..N:           (scalar tail)
//       ;;       C[m,n] += aVal * B[k,n]
//     )
//   )

function buildModule() {
  // Type codes
  const I32 = 0x7f, F32 = 0x7d, V128 = 0x7b, VOID = 0x40, FUNC_TYPE = 0x60;

  // Opcode builder — pushes bytes to the code array
  const code = [];
  const $ = (...b) => code.push(...b);

  // Opcodes
  const BLOCK = 0x02, LOOP = 0x03, END = 0x0b;
  const BR = 0x0c, BR_IF = 0x0d;
  const LOCAL_GET = 0x20, LOCAL_SET = 0x21;
  const I32_CONST = 0x41;
  const F32_LOAD = 0x2a, F32_STORE = 0x38;
  const I32_GE_U = 0x4f;
  const I32_ADD = 0x6a, I32_MUL = 0x6c, I32_AND = 0x71, I32_SHL = 0x74;
  const F32_ADD = 0x92, F32_MUL = 0x94;
  const SIMD_PREFIX = 0xfd;

  // Emitter helpers
  const get   = i => $(LOCAL_GET, ...uleb128(i));
  const set   = i => $(LOCAL_SET, ...uleb128(i));
  const iconst = n => $(I32_CONST, ...sleb128(n));

  // Memory ops (memarg: align as log2, offset — both uleb128)
  const fload  = () => $(F32_LOAD, 2, 0);          // align=4bytes, offset=0
  const fstore = () => $(F32_STORE, 2, 0);
  const vload  = () => $(SIMD_PREFIX, ...uleb128(0), 2, 0);   // v128.load
  const vstore = () => $(SIMD_PREFIX, ...uleb128(11), 2, 0);  // v128.store

  // SIMD ops
  const vsplat = () => $(SIMD_PREFIX, ...uleb128(19));   // f32x4.splat
  const vadd   = () => $(SIMD_PREFIX, ...uleb128(228));  // f32x4.add
  const vmul   = () => $(SIMD_PREFIX, ...uleb128(230));  // f32x4.mul

  // ── Parameter & local indices ──
  // Params:  aPtr=0, bPtr=1, cPtr=2, M=3, K=4, N=5
  // Locals:  m=6, k=7, n=8, aOff=9, bOff=10, cOff=11, N4=12, aVal=13(f32), aVec=14(v128)
  const P = { aPtr: 0, bPtr: 1, cPtr: 2, M: 3, K: 4, N: 5 };
  const L = { m: 6, k: 7, n: 8, aOff: 9, bOff: 10, cOff: 11, N4: 12, aVal: 13, aVec: 14 };

  // ── Function body ──

  // N4 = N & (-4)   →  round down to multiple of 4
  get(P.N); iconst(-4); $(I32_AND); set(L.N4);

  // m = 0
  iconst(0); set(L.m);

  // ── Outer loop: for m = 0..M ──
  $(BLOCK, VOID);                                        // block $m_break
    $(LOOP, VOID);                                       // loop $m_loop
      get(L.m); get(P.M); $(I32_GE_U); $(BR_IF, 1);    //   br_if $m_break (m >= M)

      // k = 0
      iconst(0); set(L.k);

      // ── Middle loop: for k = 0..K ──
      $(BLOCK, VOID);                                    // block $k_break
        $(LOOP, VOID);                                   // loop $k_loop
          get(L.k); get(P.K); $(I32_GE_U); $(BR_IF, 1); //   br_if $k_break (k >= K)

          // aOff = aPtr + (m * K + k) * 4
          get(P.aPtr);
          get(L.m); get(P.K); $(I32_MUL);
          get(L.k); $(I32_ADD);
          iconst(2); $(I32_SHL);
          $(I32_ADD);
          set(L.aOff);

          // aVal = f32.load(aOff)
          get(L.aOff); fload(); set(L.aVal);

          // aVec = f32x4.splat(aVal)
          get(L.aVal); vsplat(); set(L.aVec);

          // bOff = bPtr + (k * N) * 4
          get(P.bPtr);
          get(L.k); get(P.N); $(I32_MUL);
          iconst(2); $(I32_SHL);
          $(I32_ADD);
          set(L.bOff);

          // cOff = cPtr + (m * N) * 4
          get(P.cPtr);
          get(L.m); get(P.N); $(I32_MUL);
          iconst(2); $(I32_SHL);
          $(I32_ADD);
          set(L.cOff);

          // ── SIMD inner loop: n = 0..N4 step 4 ──
          iconst(0); set(L.n);

          $(BLOCK, VOID);                                     // block $n_break
            $(LOOP, VOID);                                    // loop $n_loop
              get(L.n); get(L.N4); $(I32_GE_U); $(BR_IF, 1); //   br_if $n_break

              // v128.store(cAddr, f32x4.add(v128.load(cAddr), f32x4.mul(aVec, v128.load(bAddr))))
              // where cAddr = cOff + n*4, bAddr = bOff + n*4

              // Push store address
              get(L.cOff); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);

              // Load C[m, n:n+4]
              get(L.cOff); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);
              vload();

              // aVec * B[k, n:n+4]
              get(L.aVec);
              get(L.bOff); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);
              vload();
              vmul();

              // Accumulate
              vadd();

              // Store result
              vstore();

              // n += 4
              get(L.n); iconst(4); $(I32_ADD); set(L.n);

              $(BR, 0);                                       // continue $n_loop
            $(END);                                           // end loop $n_loop
          $(END);                                             // end block $n_break

          // ── Scalar tail: n = N4..N ──
          $(BLOCK, VOID);                                      // block $tail_break
            $(LOOP, VOID);                                     // loop $tail_loop
              get(L.n); get(P.N); $(I32_GE_U); $(BR_IF, 1);   //   br_if $tail_break

              // f32.store(cAddr, f32.add(f32.load(cAddr), f32.mul(aVal, f32.load(bAddr))))

              // Push store address
              get(L.cOff); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);

              // Load C[m,n]
              get(L.cOff); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);
              fload();

              // aVal * B[k,n]
              get(L.aVal);
              get(L.bOff); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);
              fload();
              $(F32_MUL);

              // Accumulate
              $(F32_ADD);

              // Store
              fstore();

              // n++
              get(L.n); iconst(1); $(I32_ADD); set(L.n);

              $(BR, 0);                                        // continue $tail_loop
            $(END);                                            // end loop
          $(END);                                              // end block

          // k++
          get(L.k); iconst(1); $(I32_ADD); set(L.k);

          $(BR, 0);                                            // continue $k_loop
        $(END);                                                // end loop $k_loop
      $(END);                                                  // end block $k_break

      // m++
      get(L.m); iconst(1); $(I32_ADD); set(L.m);

      $(BR, 0);                                                // continue $m_loop
    $(END);                                                    // end loop $m_loop
  $(END);                                                      // end block $m_break

  $(END);                                                      // end function

  // ── Locals declaration ──
  // 3 groups: 7 x i32, 1 x f32, 1 x v128
  const locals = [
    ...uleb128(3),           // 3 local groups
    ...uleb128(7), I32,      // locals 6-12: m, k, n, aOff, bOff, cOff, N4
    ...uleb128(1), F32,      // local 13: aVal
    ...uleb128(1), V128,     // local 14: aVec
  ];

  // ── Assemble WASM sections ──

  // Type section: one function type (i32 x6) → ()
  const typeSec = section(1, [
    ...uleb128(1),                                  // 1 type
    FUNC_TYPE,
    ...uleb128(6), I32, I32, I32, I32, I32, I32,   // 6 params
    ...uleb128(0),                                  // 0 results
  ]);

  // Import section: memory from "env"."memory"
  const importSec = section(2, [
    ...uleb128(1),                // 1 import
    ...encStr('env'),             // module
    ...encStr('memory'),          // name
    0x02,                         // memory import kind
    0x00,                         // limits: flags (no max)
    ...uleb128(1),                // initial: 1 page
  ]);

  // Function section: func 0 → type 0
  const funcSec = section(3, [
    ...uleb128(1),
    ...uleb128(0),
  ]);

  // Export section: "matmul" → func 0
  const exportSec = section(7, [
    ...uleb128(1),
    ...encStr('matmul'),
    0x00,                         // export kind: function
    ...uleb128(0),                // func index 0
  ]);

  // Code section
  const codeBody = [...locals, ...code];
  const codeSec = section(10, [
    ...uleb128(1),                        // 1 function body
    ...uleb128(codeBody.length),          // body size
    ...codeBody,
  ]);

  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d,   // magic: \0asm
    0x01, 0x00, 0x00, 0x00,   // version: 1
    ...typeSec,
    ...importSec,
    ...funcSec,
    ...exportSec,
    ...codeSec,
  ]);
}

// ─── Runtime Functions ────────────────────────────────────────────

/**
 * Initialize the WASM SIMD matmul engine.
 * Call once at startup. Synchronous — WASM module is tiny.
 *
 * @returns {boolean} true if WASM SIMD is available
 */
function init() {
  try {
    const wasmBytes = buildModule();
    wasmMemory = new WebAssembly.Memory({ initial: INITIAL_PAGES });
    memoryPages = INITIAL_PAGES;
    const module = new WebAssembly.Module(wasmBytes);
    wasmInstance = new WebAssembly.Instance(module, { env: { memory: wasmMemory } });
    weightBumpPtr = SCRATCH_SIZE;
    _supported = true;

    // Self-test: 2x3 @ 3x4 = 2x4
    const ok = selfTest();
    if (!ok) {
      console.warn('  [WASM-SIMD] Self-test FAILED — falling back to JS');
      _supported = false;
    }

    return _supported;
  } catch (e) {
    console.warn(`  [WASM-SIMD] Not available: ${e.message}`);
    _supported = false;
    return false;
  }
}

/**
 * Self-test: verify WASM matmul matches JS reference.
 */
function selfTest() {
  const M = 2, K = 3, N = 5;
  // A = [[1,2,3],[4,5,6]], B = identity-ish 3x5
  const A = new Float32Array([1, 2, 3, 4, 5, 6]);
  const B = new Float32Array([
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
  ]);
  const C = new Float32Array(M * N);
  const expected = new Float32Array([1, 2, 3, 0, 0, 4, 5, 6, 0, 0]);

  matmul_general(A, M, K, B, N, C);

  for (let i = 0; i < C.length; i++) {
    if (Math.abs(C[i] - expected[i]) > 1e-5) return false;
  }
  return true;
}

/**
 * Ensure WASM memory has at least `neededBytes` bytes.
 */
function ensureMemory(neededBytes) {
  const currentBytes = memoryPages * 65536;
  if (neededBytes > currentBytes) {
    const additional = Math.ceil((neededBytes - currentBytes) / 65536);
    wasmMemory.grow(additional);
    memoryPages += additional;
  }
}

/**
 * Allocate a float32 weight matrix in WASM memory.
 * Persistent — survives across matmul calls. Never freed.
 *
 * @param {Float32Array} data — weight data
 * @returns {number} WASM memory pointer (byte offset)
 */
function allocWeights(data) {
  const byteSize = data.length * 4;
  // Align to 16 bytes for SIMD
  const aligned = (weightBumpPtr + 15) & ~15;
  ensureMemory(aligned + byteSize);
  new Float32Array(wasmMemory.buffer, aligned, data.length).set(data);
  weightBumpPtr = aligned + byteSize;
  return aligned;
}

/**
 * General matmul: C[M,N] += A[M,K] × B[K,N]
 * Copies both A and B to WASM scratch. Use matmulCached for weight-cached path.
 *
 * @param {Float32Array} aData
 * @param {number} M
 * @param {number} K
 * @param {Float32Array} bData
 * @param {number} N
 * @param {Float32Array} outData — output buffer, filled with result
 */
function matmul_general(aData, M, K, bData, N, outData) {
  const aBytes = M * K * 4;
  const bBytes = K * N * 4;
  const cBytes = M * N * 4;
  const totalBytes = aBytes + bBytes + cBytes;

  // Use scratch if fits, else use temporary region after weights
  let basePtr;
  if (totalBytes <= SCRATCH_SIZE) {
    basePtr = 0;
  } else {
    // Allocate after weight region (temporary, not bumped)
    basePtr = (weightBumpPtr + 15) & ~15;
    ensureMemory(basePtr + totalBytes);
  }

  const aPtr = basePtr;
  const bPtr = basePtr + aBytes;
  const cPtr = basePtr + aBytes + bBytes;

  new Float32Array(wasmMemory.buffer, aPtr, M * K).set(aData);
  new Float32Array(wasmMemory.buffer, bPtr, K * N).set(bData);
  new Float32Array(wasmMemory.buffer, cPtr, M * N).fill(0);

  wasmInstance.exports.matmul(aPtr, bPtr, cPtr, M, K, N);

  outData.set(new Float32Array(wasmMemory.buffer, cPtr, M * N));
}

/**
 * Cached matmul: out[M,N] = x[M,K] × W[K,N] + bias[N]
 * Weight matrix W is already in WASM memory at wPtr (from allocWeights).
 * Only copies the small activation vector per call.
 *
 * @param {Float32Array} xData — input [M, K]
 * @param {number} M
 * @param {number} K
 * @param {number} wPtr — WASM pointer to weight data [K, N]
 * @param {number} N
 * @param {Float32Array|null} biasData — optional bias [N]
 * @param {Float32Array} outData — output buffer [M, N]
 */
function matmulCached(xData, M, K, wPtr, N, biasData, outData) {
  const xBytes = M * K * 4;
  // Align output to 16 bytes
  const outOffset = (xBytes + 15) & ~15;
  const outBytes = M * N * 4;
  const scratchNeeded = outOffset + outBytes;

  // Use scratch if fits, else allocate temporary region above weights
  let xPtr, cPtr;
  if (scratchNeeded <= SCRATCH_SIZE) {
    xPtr = 0;
    cPtr = outOffset;
  } else {
    const basePtr = (weightBumpPtr + 15) & ~15;
    ensureMemory(basePtr + scratchNeeded);
    xPtr = basePtr;
    cPtr = basePtr + outOffset;
  }

  // Copy activation to WASM memory
  new Float32Array(wasmMemory.buffer, xPtr, M * K).set(xData);

  // Initialize output: bias (per row) or zeros
  const outView = new Float32Array(wasmMemory.buffer, cPtr, M * N);
  if (biasData) {
    for (let m = 0; m < M; m++) {
      outView.set(biasData, m * N);
    }
  } else {
    outView.fill(0);
  }

  // Run WASM SIMD matmul — accumulates into C
  wasmInstance.exports.matmul(xPtr, wPtr, cPtr, M, K, N);

  // Copy result back
  outData.set(new Float32Array(wasmMemory.buffer, cPtr, M * N));
}

/**
 * Reset weight allocator. Call when loading a new model.
 */
function resetWeights() {
  weightBumpPtr = SCRATCH_SIZE;
}

/**
 * Get memory usage stats.
 */
function getStats() {
  return {
    supported: _supported,
    totalMemoryMB: (memoryPages * 65536 / (1024 * 1024)).toFixed(1),
    weightsMB: ((weightBumpPtr - SCRATCH_SIZE) / (1024 * 1024)).toFixed(1),
    scratchMB: (SCRATCH_SIZE / (1024 * 1024)).toFixed(1),
  };
}

module.exports = {
  init,
  allocWeights,
  matmul_general,
  matmulCached,
  resetWeights,
  getStats,
  get supported() { return _supported; },
};
