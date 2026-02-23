/**
 * PureBee — WASM Q4_0/Q4_1 Matrix-Vector Multiply Kernel
 *
 * Computes y = W @ x directly on quantized Q4 nibble data,
 * skipping dequantization and transpose entirely.
 *
 * The WASM binary is built programmatically — zero external dependencies.
 *
 * Q4_0 block (18 bytes / 32 values): [f16 scale][16 nibble bytes]
 * Q4_1 block (20 bytes / 32 values): [f16 delta][f16 min][16 nibble bytes]
 */

'use strict';

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

// ─── Opcodes ──────────────────────────────────────────────────────

const I32 = 0x7f, F32 = 0x7d, VOID = 0x40, FUNC_TYPE = 0x60;
const BLOCK = 0x02, LOOP = 0x03, IF = 0x04, ELSE = 0x05, END = 0x0b;
const BR = 0x0c, BR_IF = 0x0d;
const LOCAL_GET = 0x20, LOCAL_SET = 0x21;
const I32_CONST = 0x41, F32_CONST = 0x43;
const I32_LOAD8_U = 0x2d, I32_LOAD16_U = 0x2f;
const F32_LOAD = 0x2a, F32_STORE = 0x38;
const I32_EQZ = 0x45, I32_GE_U = 0x4f;
const I32_ADD = 0x6a, I32_SUB = 0x6b, I32_MUL = 0x6c;
const I32_AND = 0x71, I32_OR = 0x72;
const I32_SHL = 0x74, I32_SHR_U = 0x76;
const F32_ADD = 0x92, F32_MUL = 0x94;
const F32_CONVERT_I32_S = 0xb2;
const F32_REINTERPRET_I32 = 0xbe;

// ─── Code Emitter Helpers ─────────────────────────────────────────

function createEmitter() {
  const code = [];
  const $ = (...b) => code.push(...b);
  const get = i => $(LOCAL_GET, ...uleb128(i));
  const set = i => $(LOCAL_SET, ...uleb128(i));
  const iconst = n => $(I32_CONST, ...sleb128(n));
  const fconst = v => {
    $(F32_CONST);
    const buf = new ArrayBuffer(4);
    new Float32Array(buf)[0] = v;
    const bytes = new Uint8Array(buf);
    $(bytes[0], bytes[1], bytes[2], bytes[3]);
  };
  return { code, $, get, set, iconst, fconst };
}

// ─── f16→f32 conversion sequence ──────────────────────────────────
// Emits code that reads f16 bits from a local, leaves f32 on stack.
// Handles exp=0 as zero (sufficient for Q4 scales).

function emitF16toF32(e, scaleBitsLocal, expLocal) {
  const { $, get, set, iconst, fconst } = e;

  // exp = (scaleBits >> 10) & 0x1F
  get(scaleBitsLocal); iconst(10); $(I32_SHR_U); iconst(0x1F); $(I32_AND);
  set(expLocal);

  get(expLocal); $(I32_EQZ);
  $(IF, F32);
    fconst(0.0);
  $(ELSE);
    // sign << 31
    get(scaleBitsLocal); iconst(15); $(I32_SHR_U); iconst(31); $(I32_SHL);
    // | ((exp + 112) << 23)
    get(expLocal); iconst(112); $(I32_ADD); iconst(23); $(I32_SHL);
    $(I32_OR);
    // | ((scaleBits & 0x3FF) << 13)
    get(scaleBitsLocal); iconst(0x3FF); $(I32_AND); iconst(13); $(I32_SHL);
    $(I32_OR);
    $(F32_REINTERPRET_I32);
  $(END);
}

// ─── Build Q4_0 SIMD matvec function body ───────────────────────
// q4_0_matvec(xPtr, rawPtr, outPtr, N, K)
//
// SIMD strategy: load 16 nibble bytes as v128, shuffle to spread
// 4 bytes into i32x4 lanes, extract lo/hi nibbles, convert to f32x4,
// multiply by x vector, accumulate. Fully unrolled inner loop.

function buildQ4_0Body() {
  const e = createEmitter();
  const { $, get, set, iconst, fconst } = e;

  // SIMD helpers
  const simd = (op) => { $(SIMD_PREFIX); for (const b of uleb128(op)) $(b); };
  const vload = (align, off) => { simd(0); $(align); $(off || 0); };
  const vconst16 = (bytes) => { simd(12); for (let i = 0; i < 16; i++) $(bytes[i] || 0); };
  const vand = () => simd(78);
  const i32x4_shr_u = () => simd(173);
  const i32x4_sub = () => simd(177);
  const f32x4_convert = () => simd(250);  // f32x4.convert_i32x4_s
  const f32x4_splat = () => simd(19);
  const vadd = () => simd(228);  // f32x4.add
  const vmul = () => simd(230);  // f32x4.mul
  const shuffle = (lanes) => { simd(13); for (const l of lanes) $(l); };
  const extract_f32 = (lane) => { simd(31); $(lane); };

  // Params: xPtr=0, rawPtr=1, outPtr=2, N=3, K=4
  const P = { xPtr: 0, rawPtr: 1, outPtr: 2, N: 3, K: 4 };
  // Locals: 10 i32, 2 f32, 7 v128
  const L = {
    n: 5, b: 6, blocksPerRow: 7, bytesPerRow: 8,
    rowOff: 9, bOff: 10, dOff: 11, xBase: 12,
    scaleBits: 13, exp: 14,
    scale: 15, sum: 16,
    acc: 17, blockAcc: 18, nibbles: 19, zero: 20, mask0F: 21, sub8: 22, tmp: 23,
  };

  // ── Initialize SIMD constants ──
  vconst16([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]); set(L.zero);
  vconst16([0x0F,0,0,0, 0x0F,0,0,0, 0x0F,0,0,0, 0x0F,0,0,0]); set(L.mask0F);
  vconst16([8,0,0,0, 8,0,0,0, 8,0,0,0, 8,0,0,0]); set(L.sub8);

  // blocksPerRow = K >> 5
  get(P.K); iconst(5); $(I32_SHR_U); set(L.blocksPerRow);
  // bytesPerRow = blocksPerRow * 18
  get(L.blocksPerRow); iconst(18); $(I32_MUL); set(L.bytesPerRow);

  iconst(0); set(L.n);

  // ── outer loop: for n = 0..N-1 ──
  $(BLOCK, VOID);
    $(LOOP, VOID);
      get(L.n); get(P.N); $(I32_GE_U); $(BR_IF, 1);

      get(L.zero); set(L.acc);

      // rowOff = rawPtr + n * bytesPerRow
      get(P.rawPtr); get(L.n); get(L.bytesPerRow); $(I32_MUL); $(I32_ADD);
      set(L.rowOff);

      iconst(0); set(L.b);

      // ── block loop: for b = 0..blocksPerRow-1 ──
      $(BLOCK, VOID);
        $(LOOP, VOID);
          get(L.b); get(L.blocksPerRow); $(I32_GE_U); $(BR_IF, 1);

          // bOff = rowOff + b * 18
          get(L.rowOff); get(L.b); iconst(18); $(I32_MUL); $(I32_ADD);
          set(L.bOff);

          // scale = f16_to_f32(load16_u(bOff))
          get(L.bOff); $(I32_LOAD16_U, 0, 0); set(L.scaleBits);
          emitF16toF32(e, L.scaleBits, L.exp);
          set(L.scale);

          // dOff = bOff + 2
          get(L.bOff); iconst(2); $(I32_ADD); set(L.dOff);

          // xBase = xPtr + b * 128  (b * 32 values * 4 bytes/value)
          get(P.xPtr); get(L.b); iconst(7); $(I32_SHL); $(I32_ADD);
          set(L.xBase);

          get(L.zero); set(L.blockAcc);

          // Load all 16 nibble bytes as v128
          get(L.dOff); vload(0, 0); set(L.nibbles);

          // ── Unrolled: 4 groups of 4 bytes ──
          // Each group processes 8 values (4 lo nibbles + 4 hi nibbles)
          for (const g of [0, 4, 8, 12]) {
            // Spread bytes g..g+3 into i32x4 lanes (zero-extend each byte)
            get(L.nibbles); get(L.zero);
            shuffle([g, 16, 16, 16,  g+1, 16, 16, 16,  g+2, 16, 16, 16,  g+3, 16, 16, 16]);
            set(L.tmp);

            // Low nibbles: AND 0x0F, SUB 8, convert to f32x4
            get(L.tmp); get(L.mask0F); vand();
            get(L.sub8); i32x4_sub();
            f32x4_convert();
            // Multiply by x[g..g+3]
            get(L.xBase); vload(2, g * 4);
            vmul();
            get(L.blockAcc); vadd(); set(L.blockAcc);

            // High nibbles: SHR_U 4, SUB 8, convert to f32x4
            get(L.tmp); iconst(4); i32x4_shr_u();
            get(L.sub8); i32x4_sub();
            f32x4_convert();
            // Multiply by x[g+16..g+19]
            get(L.xBase); vload(2, (g + 16) * 4);
            vmul();
            get(L.blockAcc); vadd(); set(L.blockAcc);
          }

          // acc += blockAcc * scale
          get(L.blockAcc);
          get(L.scale); f32x4_splat();
          vmul();
          get(L.acc); vadd(); set(L.acc);

          get(L.b); iconst(1); $(I32_ADD); set(L.b);
          $(BR, 0);
        $(END);
      $(END);

      // ── Horizontal sum: acc → scalar sum ──
      get(L.acc); get(L.acc);
      shuffle([8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7]);
      set(L.tmp);
      get(L.acc); get(L.tmp); vadd(); set(L.tmp);
      get(L.tmp); get(L.tmp);
      shuffle([4,5,6,7, 0,1,2,3, 0,1,2,3, 0,1,2,3]);
      get(L.tmp); vadd();
      extract_f32(0);
      set(L.sum);

      // store output[n] = sum
      get(P.outPtr); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);
      get(L.sum);
      $(F32_STORE, 2, 0);

      get(L.n); iconst(1); $(I32_ADD); set(L.n);
      $(BR, 0);
    $(END);
  $(END);

  $(END); // end function

  const locals = [
    ...uleb128(3),           // 3 groups
    ...uleb128(10), I32,     // n, b, blocksPerRow, bytesPerRow, rowOff, bOff, dOff, xBase, scaleBits, exp
    ...uleb128(2), F32,      // scale, sum
    ...uleb128(7), V128,     // acc, blockAcc, nibbles, zero, mask0F, sub8, tmp
  ];

  return [...locals, ...e.code];
}

// ─── Build Q4_1 SIMD matvec function body ───────────────────────
// q4_1_matvec(xPtr, rawPtr, outPtr, N, K)
//
// Q4_1: sum += delta * nibbleSum + min * xBlockSum
// SIMD: two v128 accumulators — nibAcc (x*nibble) and xAcc (sum of x)

function buildQ4_1Body() {
  const e = createEmitter();
  const { $, get, set, iconst, fconst } = e;

  // SIMD helpers
  const simd = (op) => { $(SIMD_PREFIX); for (const b of uleb128(op)) $(b); };
  const vload = (align, off) => { simd(0); $(align); $(off || 0); };
  const vconst16 = (bytes) => { simd(12); for (let i = 0; i < 16; i++) $(bytes[i] || 0); };
  const vand = () => simd(78);
  const i32x4_shr_u = () => simd(173);
  const f32x4_convert = () => simd(250);  // f32x4.convert_i32x4_s
  const f32x4_splat = () => simd(19);
  const vadd = () => simd(228);  // f32x4.add
  const vmul = () => simd(230);  // f32x4.mul
  const shuffle = (lanes) => { simd(13); for (const l of lanes) $(l); };
  const extract_f32 = (lane) => { simd(31); $(lane); };

  const P = { xPtr: 0, rawPtr: 1, outPtr: 2, N: 3, K: 4 };
  // Locals: 10 i32, 3 f32, 9 v128
  const L = {
    n: 5, b: 6, blocksPerRow: 7, bytesPerRow: 8,
    rowOff: 9, bOff: 10, dOff: 11, xBase: 12,
    scaleBits: 13, exp: 14,
    delta: 15, minVal: 16, sum: 17,
    acc: 18, nibAcc: 19, xAcc: 20, nibbles: 21,
    zero: 22, mask0F: 23, tmp: 24, xVec: 25, tmp2: 26,
  };

  // ── Initialize SIMD constants ──
  vconst16([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]); set(L.zero);
  vconst16([0x0F,0,0,0, 0x0F,0,0,0, 0x0F,0,0,0, 0x0F,0,0,0]); set(L.mask0F);

  // blocksPerRow = K >> 5
  get(P.K); iconst(5); $(I32_SHR_U); set(L.blocksPerRow);
  // bytesPerRow = blocksPerRow * 20
  get(L.blocksPerRow); iconst(20); $(I32_MUL); set(L.bytesPerRow);

  iconst(0); set(L.n);

  // ── outer loop: for n = 0..N-1 ──
  $(BLOCK, VOID);
    $(LOOP, VOID);
      get(L.n); get(P.N); $(I32_GE_U); $(BR_IF, 1);

      get(L.zero); set(L.acc);

      get(P.rawPtr); get(L.n); get(L.bytesPerRow); $(I32_MUL); $(I32_ADD);
      set(L.rowOff);

      iconst(0); set(L.b);

      // ── block loop ──
      $(BLOCK, VOID);
        $(LOOP, VOID);
          get(L.b); get(L.blocksPerRow); $(I32_GE_U); $(BR_IF, 1);

          // bOff = rowOff + b * 20
          get(L.rowOff); get(L.b); iconst(20); $(I32_MUL); $(I32_ADD);
          set(L.bOff);

          // delta = f16(load16_u(bOff))
          get(L.bOff); $(I32_LOAD16_U, 0, 0); set(L.scaleBits);
          emitF16toF32(e, L.scaleBits, L.exp);
          set(L.delta);

          // min = f16(load16_u(bOff + 2))
          get(L.bOff); iconst(2); $(I32_ADD); $(I32_LOAD16_U, 0, 0); set(L.scaleBits);
          emitF16toF32(e, L.scaleBits, L.exp);
          set(L.minVal);

          // dOff = bOff + 4
          get(L.bOff); iconst(4); $(I32_ADD); set(L.dOff);
          // xBase = xPtr + b * 128
          get(P.xPtr); get(L.b); iconst(7); $(I32_SHL); $(I32_ADD);
          set(L.xBase);

          get(L.zero); set(L.nibAcc);
          get(L.zero); set(L.xAcc);

          // Load all 16 nibble bytes as v128
          get(L.dOff); vload(0, 0); set(L.nibbles);

          // ── Unrolled: 4 groups of 4 bytes ──
          for (const g of [0, 4, 8, 12]) {
            // Spread bytes g..g+3 into i32x4 lanes
            get(L.nibbles); get(L.zero);
            shuffle([g, 16, 16, 16,  g+1, 16, 16, 16,  g+2, 16, 16, 16,  g+3, 16, 16, 16]);
            set(L.tmp);

            // ── Low nibbles: x[g..g+3] * nibble, and sum x ──
            // nibble values (unsigned 0-15)
            get(L.tmp); get(L.mask0F); vand();
            f32x4_convert();  // nibble → f32x4
            // Load x[g..g+3]
            get(L.xBase); vload(2, g * 4);
            set(L.xVec);
            // nibAcc += xVec * nibble
            get(L.xVec);
            vmul();
            get(L.nibAcc); vadd(); set(L.nibAcc);
            // xAcc += xVec
            get(L.xAcc); get(L.xVec); vadd(); set(L.xAcc);

            // ── High nibbles: x[g+16..g+19] * nibble, and sum x ──
            get(L.tmp); iconst(4); i32x4_shr_u();
            f32x4_convert();
            // Load x[g+16..g+19]
            get(L.xBase); vload(2, (g + 16) * 4);
            set(L.xVec);
            get(L.xVec);
            vmul();
            get(L.nibAcc); vadd(); set(L.nibAcc);
            get(L.xAcc); get(L.xVec); vadd(); set(L.xAcc);
          }

          // acc += delta * nibAcc + min * xAcc
          get(L.nibAcc);
          get(L.delta); f32x4_splat();
          vmul();
          get(L.xAcc);
          get(L.minVal); f32x4_splat();
          vmul();
          vadd();
          get(L.acc); vadd(); set(L.acc);

          get(L.b); iconst(1); $(I32_ADD); set(L.b);
          $(BR, 0);
        $(END);
      $(END);

      // ── Horizontal sum ──
      get(L.acc); get(L.acc);
      shuffle([8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7]);
      set(L.tmp);
      get(L.acc); get(L.tmp); vadd(); set(L.tmp);
      get(L.tmp); get(L.tmp);
      shuffle([4,5,6,7, 0,1,2,3, 0,1,2,3, 0,1,2,3]);
      get(L.tmp); vadd();
      extract_f32(0);
      set(L.sum);

      get(P.outPtr); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);
      get(L.sum);
      $(F32_STORE, 2, 0);

      get(L.n); iconst(1); $(I32_ADD); set(L.n);
      $(BR, 0);
    $(END);
  $(END);

  $(END);

  const locals = [
    ...uleb128(3),           // 3 groups
    ...uleb128(10), I32,     // n, b, blocksPerRow, bytesPerRow, rowOff, bOff, dOff, xBase, scaleBits, exp
    ...uleb128(3), F32,      // delta, minVal, sum
    ...uleb128(9), V128,     // acc, nibAcc, xAcc, nibbles, zero, mask0F, tmp, xVec, tmp2
  ];

  return [...locals, ...e.code];
}

// ─── Build f32 SIMD batch_dot function body ──────────────────────
// batch_dot(xPtr, wPtr, outPtr, N, K)
//
// Computes N dot products: out[n] = sum_k(x[k] * W[n*K + k])
// W is [N, K] row-major float32. Uses f32x4 SIMD on K dimension.
// K must be a multiple of 4.

const V128 = 0x7b;
const SIMD_PREFIX = 0xfd;

function buildBatchDotBody() {
  const e = createEmitter();
  const { $, get, set, iconst, fconst } = e;

  // SIMD helpers
  const vload  = (align) => $(SIMD_PREFIX, ...uleb128(0), align || 2, 0);   // v128.load
  const vconst = (bytes) => { $(SIMD_PREFIX, ...uleb128(12)); for (let i = 0; i < 16; i++) $(bytes[i] || 0); };
  const vadd   = () => $(SIMD_PREFIX, ...uleb128(228));  // f32x4.add
  const vmul   = () => $(SIMD_PREFIX, ...uleb128(230));  // f32x4.mul
  const shuffle = (lanes) => { $(SIMD_PREFIX, ...uleb128(13)); for (const l of lanes) $(l); };
  const extract_f32 = (lane) => $(SIMD_PREFIX, ...uleb128(31), lane); // f32x4.extract_lane

  // Params: xPtr=0, wPtr=1, outPtr=2, N=3, K=4
  // Locals (i32): n=5, k=6, K4=7, wOff=8, kBytes=9
  // Locals (f32): sum=10
  // Locals (v128): acc=11, tmp=12
  const P = { xPtr: 0, wPtr: 1, outPtr: 2, N: 3, K: 4 };
  const L = { n: 5, k: 6, K4: 7, wOff: 8, kBytes: 9, sum: 10, acc: 11, tmp: 12 };

  // K4 = K & (-4)
  get(P.K); iconst(-4); $(I32_AND); set(L.K4);

  // kBytes = K * 4
  get(P.K); iconst(2); $(I32_SHL); set(L.kBytes);

  iconst(0); set(L.n);

  // ── outer loop: for n = 0..N-1 ──
  $(BLOCK, VOID);
    $(LOOP, VOID);
      get(L.n); get(P.N); $(I32_GE_U); $(BR_IF, 1);

      // acc = f32x4(0,0,0,0)
      vconst([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]);
      set(L.acc);

      // wOff = wPtr + n * kBytes
      get(P.wPtr); get(L.n); get(L.kBytes); $(I32_MUL); $(I32_ADD);
      set(L.wOff);

      iconst(0); set(L.k);

      // ── SIMD inner loop: k = 0..K4 step 4 ──
      $(BLOCK, VOID);
        $(LOOP, VOID);
          get(L.k); get(L.K4); $(I32_GE_U); $(BR_IF, 1);

          // acc += f32x4.mul(x[k..k+3], W[wOff+k..+3])
          get(L.acc);
          get(P.xPtr); get(L.k); iconst(2); $(I32_SHL); $(I32_ADD);
          vload(2);
          get(L.wOff); get(L.k); iconst(2); $(I32_SHL); $(I32_ADD);
          vload(2);
          vmul();
          vadd();
          set(L.acc);

          get(L.k); iconst(4); $(I32_ADD); set(L.k);
          $(BR, 0);
        $(END);
      $(END);

      // ── Horizontal sum of acc → sum local ──
      // acc = [a, b, c, d]
      // swap halves: [c, d, a, b]
      get(L.acc); get(L.acc);
      shuffle([8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7]);
      set(L.tmp);
      // [a+c, b+d, ?, ?]
      get(L.acc); get(L.tmp); vadd(); set(L.tmp);
      // swap low pair: [b+d, a+c, ?, ?]
      get(L.tmp); get(L.tmp);
      shuffle([4,5,6,7, 0,1,2,3, 0,1,2,3, 0,1,2,3]);
      // [a+b+c+d, ?, ?, ?]
      get(L.tmp); vadd();
      // extract lane 0 → f32
      extract_f32(0);
      set(L.sum);

      // ── Scalar tail: k = K4..K ──
      $(BLOCK, VOID);
        $(LOOP, VOID);
          get(L.k); get(P.K); $(I32_GE_U); $(BR_IF, 1);

          // sum += x[k] * W[wOff + k*4]
          get(L.sum);
          get(P.xPtr); get(L.k); iconst(2); $(I32_SHL); $(I32_ADD);
          $(F32_LOAD, 2, 0);
          get(L.wOff); get(L.k); iconst(2); $(I32_SHL); $(I32_ADD);
          $(F32_LOAD, 2, 0);
          $(F32_MUL);
          $(F32_ADD);
          set(L.sum);

          get(L.k); iconst(1); $(I32_ADD); set(L.k);
          $(BR, 0);
        $(END);
      $(END);

      // f32.store(outPtr + n*4, sum)
      get(P.outPtr); get(L.n); iconst(2); $(I32_SHL); $(I32_ADD);
      get(L.sum);
      $(F32_STORE, 2, 0);

      get(L.n); iconst(1); $(I32_ADD); set(L.n);
      $(BR, 0);
    $(END);
  $(END);

  $(END); // end function

  const locals = [
    ...uleb128(3),          // 3 local groups
    ...uleb128(5), I32,     // n, k, K4, wOff, kBytes
    ...uleb128(1), F32,     // sum
    ...uleb128(2), V128,    // acc, tmp
  ];

  return [...locals, ...e.code];
}

// ─── Build WASM Module ────────────────────────────────────────────

function buildQ4Module() {
  const q4_0_body = buildQ4_0Body();
  const q4_1_body = buildQ4_1Body();
  const batch_dot_body = buildBatchDotBody();

  // Type section: one type (i32 x5) → ()
  const typeSec = section(1, [
    ...uleb128(1),
    FUNC_TYPE,
    ...uleb128(5), I32, I32, I32, I32, I32,
    ...uleb128(0),
  ]);

  // Import: memory from "env"
  const importSec = section(2, [
    ...uleb128(1),
    ...encStr('env'), ...encStr('memory'),
    0x02, 0x00, ...uleb128(1),
  ]);

  // Function section: 3 functions, all type 0
  const funcSec = section(3, [
    ...uleb128(3),
    ...uleb128(0),
    ...uleb128(0),
    ...uleb128(0),
  ]);

  // Export section
  const exportSec = section(7, [
    ...uleb128(3),
    ...encStr('q4_0_matvec'), 0x00, ...uleb128(0),
    ...encStr('q4_1_matvec'), 0x00, ...uleb128(1),
    ...encStr('batch_dot'), 0x00, ...uleb128(2),
  ]);

  // Code section: 3 function bodies
  const codeSec = section(10, [
    ...uleb128(3),
    ...uleb128(q4_0_body.length), ...q4_0_body,
    ...uleb128(q4_1_body.length), ...q4_1_body,
    ...uleb128(batch_dot_body.length), ...batch_dot_body,
  ]);

  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d,
    0x01, 0x00, 0x00, 0x00,
    ...typeSec,
    ...importSec,
    ...funcSec,
    ...exportSec,
    ...codeSec,
  ]);
}

// ─── Runtime ──────────────────────────────────────────────────────

let wasmInstance = null;
let wasmMemory = null;
let memoryPages = 0;
let _supported = false;

const INITIAL_PAGES = 256; // 16MB

function ensureMemory(neededBytes) {
  const currentBytes = memoryPages * 65536;
  if (neededBytes > currentBytes) {
    const additional = Math.ceil((neededBytes - currentBytes) / 65536);
    wasmMemory.grow(additional);
    memoryPages += additional;
  }
}

/**
 * Initialize WASM Q4 matvec engine. Synchronous.
 * @returns {boolean} true if available
 */
function init() {
  try {
    const wasmBytes = buildQ4Module();
    wasmMemory = new WebAssembly.Memory({ initial: INITIAL_PAGES });
    memoryPages = INITIAL_PAGES;
    const module = new WebAssembly.Module(wasmBytes);
    wasmInstance = new WebAssembly.Instance(module, { env: { memory: wasmMemory } });
    _supported = true;

    const ok = selfTest();
    if (!ok) {
      console.warn('  [WASM-Q4] Self-test FAILED — falling back to JS');
      _supported = false;
    }

    return _supported;
  } catch (e) {
    console.warn(`  [WASM-Q4] Not available: ${e.message}`);
    _supported = false;
    return false;
  }
}

/**
 * Self-test: verify Q4_0 matvec against JS reference.
 */
function selfTest() {
  // Create a simple Q4_0 test: 2 rows, K=32 (1 block per row)
  const K = 32, N = 2;

  // Input vector: x[i] = 1.0 for all i
  const x = new Float32Array(K).fill(1.0);

  // Build raw Q4_0 data: 2 rows × 1 block × 18 bytes = 36 bytes
  const raw = new Uint8Array(36);

  // Row 0: scale = 1.0 (f16 = 0x3C00), all nibbles = 9 (value = 9-8 = 1)
  // So each value = 1.0 * 1 = 1.0, dot product = 32 * 1.0 = 32.0
  raw[0] = 0x00; raw[1] = 0x3C; // f16 1.0
  for (let j = 2; j < 18; j++) raw[j] = 0x99; // nibble 9 in both positions

  // Row 1: scale = 2.0 (f16 = 0x4000), all nibbles = 10 (value = 10-8 = 2)
  // So each value = 2.0 * 2 = 4.0, dot product = 32 * 4.0 = 128.0
  raw[18] = 0x00; raw[19] = 0x40; // f16 2.0
  for (let j = 20; j < 36; j++) raw[j] = 0xAA; // nibble A in both positions

  const result = q4_0_matvec(x, raw, N, K);

  const ok0 = Math.abs(result[0] - 32.0) < 0.1;
  const ok1 = Math.abs(result[1] - 128.0) < 0.1;

  if (!ok0 || !ok1) {
    console.warn(`  [WASM-Q4] Self-test: expected [32, 128], got [${result[0]}, ${result[1]}]`);
    return false;
  }

  return true;
}

/**
 * Q4_0 matrix-vector multiply using WASM.
 * Computes output[n] = sum_k(W[n,k] * x[k]) where W is Q4_0 encoded.
 *
 * @param {Float32Array} x — input vector [K]
 * @param {Uint8Array|Buffer} rawBuf — raw Q4_0 data [N * blocksPerRow * 18]
 * @param {number} N — number of output elements (rows)
 * @param {number} K — number of input elements (cols, must be multiple of 32)
 * @returns {Float32Array} — output vector [N]
 */
function q4_0_matvec(x, rawBuf, N, K) {
  if (!wasmInstance && !init()) {
    throw new Error('WASM Q4 not available');
  }

  const xBytes = K * 4;
  const blocksPerRow = K >>> 5;
  const rawBytes = N * blocksPerRow * 18;
  const outBytes = N * 4;

  // Memory layout: [x (16-aligned)][raw (16-aligned)][out (16-aligned)]
  const xPtr = 0;
  const rawPtr = (xBytes + 15) & ~15;
  const outPtr = (rawPtr + rawBytes + 15) & ~15;
  const totalNeeded = outPtr + outBytes;

  ensureMemory(totalNeeded);

  // Copy input data to WASM memory
  new Float32Array(wasmMemory.buffer, xPtr, K).set(x);
  const rawSrc = rawBuf instanceof Uint8Array ? rawBuf :
    new Uint8Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength);
  new Uint8Array(wasmMemory.buffer, rawPtr, rawBytes).set(rawSrc.subarray(0, rawBytes));

  // Call WASM
  wasmInstance.exports.q4_0_matvec(xPtr, rawPtr, outPtr, N, K);

  // Read output
  const output = new Float32Array(N);
  output.set(new Float32Array(wasmMemory.buffer, outPtr, N));
  return output;
}

/**
 * Q4_1 matrix-vector multiply using WASM.
 *
 * @param {Float32Array} x — input vector [K]
 * @param {Uint8Array|Buffer} rawBuf — raw Q4_1 data [N * blocksPerRow * 20]
 * @param {number} N — number of output elements (rows)
 * @param {number} K — number of input elements (cols, must be multiple of 32)
 * @returns {Float32Array} — output vector [N]
 */
function q4_1_matvec(x, rawBuf, N, K) {
  if (!wasmInstance && !init()) {
    throw new Error('WASM Q4 not available');
  }

  const xBytes = K * 4;
  const blocksPerRow = K >>> 5;
  const rawBytes = N * blocksPerRow * 20;
  const outBytes = N * 4;

  const xPtr = 0;
  const rawPtr = (xBytes + 15) & ~15;
  const outPtr = (rawPtr + rawBytes + 15) & ~15;
  const totalNeeded = outPtr + outBytes;

  ensureMemory(totalNeeded);

  new Float32Array(wasmMemory.buffer, xPtr, K).set(x);
  const rawSrc = rawBuf instanceof Uint8Array ? rawBuf :
    new Uint8Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength);
  new Uint8Array(wasmMemory.buffer, rawPtr, rawBytes).set(rawSrc.subarray(0, rawBytes));

  wasmInstance.exports.q4_1_matvec(xPtr, rawPtr, outPtr, N, K);

  const output = new Float32Array(N);
  output.set(new Float32Array(wasmMemory.buffer, outPtr, N));
  return output;
}

/**
 * Chunked f32 LM head: compute logits = embedding @ hidden.
 * Processes vocabSize rows in chunks that fit in WASM memory.
 * No transpose needed — works directly on row-major embedding.
 *
 * @param {Float32Array} hidden — normalized hidden state [dim]
 * @param {Float32Array} embData — embedding weights [vocabSize * dim] row-major
 * @param {number} vocabSize — number of vocabulary tokens (rows)
 * @param {number} dim — embedding dimension (cols)
 * @returns {Float32Array} — logits [vocabSize]
 */
function lmHead(hidden, embData, vocabSize, dim) {
  if (!wasmInstance && !init()) {
    throw new Error('WASM Q4 not available');
  }

  const CHUNK = 1024;  // rows per chunk: 1024 * 2048 * 4 = 8MB
  const logits = new Float32Array(vocabSize);

  const xBytes = dim * 4;
  const xPtr = 0;
  const wPtr = (xBytes + 15) & ~15;

  // Copy hidden vector once
  ensureMemory(wPtr + CHUNK * dim * 4 + CHUNK * 4 + 16);
  new Float32Array(wasmMemory.buffer, xPtr, dim).set(hidden);

  for (let v0 = 0; v0 < vocabSize; v0 += CHUNK) {
    const chunkSize = Math.min(CHUNK, vocabSize - v0);
    const wBytes = chunkSize * dim * 4;
    const outPtr = (wPtr + wBytes + 15) & ~15;
    const totalNeeded = outPtr + chunkSize * 4;

    ensureMemory(totalNeeded);

    // Copy embedding chunk to WASM memory (contiguous rows, no transpose)
    new Float32Array(wasmMemory.buffer, wPtr, chunkSize * dim)
      .set(embData.subarray(v0 * dim, (v0 + chunkSize) * dim));

    // Rewrite hidden vector if memory grew (buffer detached)
    if (v0 > 0) {
      new Float32Array(wasmMemory.buffer, xPtr, dim).set(hidden);
    }

    // Call WASM SIMD batch_dot
    wasmInstance.exports.batch_dot(xPtr, wPtr, outPtr, chunkSize, dim);

    // Read chunk output
    logits.set(new Float32Array(wasmMemory.buffer, outPtr, chunkSize), v0);
  }

  return logits;
}

module.exports = {
  init,
  q4_0_matvec,
  q4_1_matvec,
  lmHead,
  get supported() { return _supported; },
};
