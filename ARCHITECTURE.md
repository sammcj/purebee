# Architecture

PureBee is four clean layers. Each layer has one job. No layer knows more than it needs to.

```
┌──────────────────────────────────────────────────────────────┐
│  RUNTIME  (llama-streaming.js)                               │
│  Model execution, KV-cache, tokenization, layer streaming    │
├──────────────────────────────────────────────────────────────┤
│  INSTRUCTION SET  (purebee.js)                               │
│  Named operations: TENSOR_MUL, SOFTMAX, RMS_NORM, ATTENTION  │
├──────────────────────────────────────────────────────────────┤
│  ENGINE  (engine.js + wasm-q4.js + wasm-simd.js)            │
│  Actual compute: WASM SIMD kernels, quantized matvec         │
├──────────────────────────────────────────────────────────────┤
│  MEMORY  (memory.js)                                         │
│  Tensor layout, named grid registry, quantized storage       │
└──────────────────────────────────────────────────────────────┘
```

The boundary matters. The instruction set doesn't know what SIMD is. The engine doesn't know what a transformer is. Correctness is testable at every layer independently.

---

## Layer 1 — Memory

The fundamental unit is a Tensor: a typed array with shape metadata.

```js
new Tensor([batch, seq, dim], Float32Array)
```

Above tensors sits `PureBeeMemory` — a named grid registry. Operations address tensors by name, not by pointer. This is not an accident. Named addressing means the instruction set reads like a spec:

```js
gpu.TENSOR_MUL('query', 'key_cache', 'scores')
gpu.SOFTMAX('scores', 'attn_weights')
```

The grid registry also handles lifetime. `gpu.FREE('tensor_name')` is explicit and auditable — no garbage collection surprises in the hot path.

Quantized storage lives here too. `QuantizedTensor` stores raw Q4_0/Q8_0 bytes alongside shape metadata. The memory layer doesn't dequantize — it just stores. Dequantization is an engine concern.

---

## Layer 2 — Engine

The engine is where math happens. It has two implementations of the same operations: JavaScript fallback and WASM SIMD fast path.

The WASM modules are built entirely in JavaScript — no compiler, no toolchain, no Emscripten. The binary is constructed byte by byte at startup:

```js
// Actual code from wasm-q4.js
function buildModule(bodyFn) {
  const bytes = [];
  bytes.push(0x00, 0x61, 0x73, 0x6d);  // magic: \0asm
  bytes.push(0x01, 0x00, 0x00, 0x00);  // version: 1
  // ... type section, import section, function section ...
  bodyFn(bytes);  // emit the actual compute kernel
  return new WebAssembly.Module(new Uint8Array(bytes));
}
```

This means the entire compute stack — including the binary it runs — is readable JavaScript. No build step. No black box.

### Q4 SIMD Kernel

The most complex piece. Q4_0 stores weights as 4-bit nibbles packed two-per-byte, with a float16 scale per block of 32 values. The naive approach unpacks nibbles in a scalar loop. The SIMD approach processes 32 values (16 bytes) per iteration using WebAssembly SIMD instructions:

```
Input: 16 bytes = 32 packed nibbles

Step 1: Load 16 bytes as v128
Step 2: i8x16.shuffle → spread 4 bytes into 4 i32 lanes
Step 3: v128.and(0x0F0F0F0F) → extract low nibbles (values 0–15)
Step 4: i32x4.shr_u(4) → extract high nibbles (values 16–31)
Step 5: i32x4.sub(8) → center around zero (quantization offset)
Step 6: f32x4.convert → float
Step 7: f32x4.mul(x_vec) → multiply by input
Step 8: Accumulate into block sum
```

Four groups unrolled per block iteration. Result: ~12 GFLOP/s from a single JavaScript process.

### Weight Caching

WASM memory is linear and fixed at allocation. Rather than copy weights in and out every forward pass, the engine maintains a bump allocator: weights are written once to WASM memory during model load and addressed by offset thereafter. Only the input vector (small: dim × 4 bytes) moves per operation.

---

## Layer 3 — Instruction Set

`purebee.js` defines the PureBee instruction set. Each instruction is a named method on the `PureBee` class:

```
GRID_ALLOC(name, shape)
GRID_WRITE(name, shape, data)
GRID_READ(name) → Tensor
TENSOR_MUL(a, b, out)
TENSOR_ADD(a, b, out)
LINEAR(input, weight, bias, out)
SOFTMAX(input, out)
LAYER_NORM(input, weight, out)
RMS_NORM(input, weight, out)
GELU(input, out)
SILU(input, out)
ATTENTION(q, k, v, mask, out)
ELEMENT_MUL(a, b, out)
SYNC()
FREE(name)
```

This is the spec. A transformer can be written using only these operations. The instruction set is hardware-agnostic by design — it describes *what*, not *how*.

`SYNC()` is a fence. Operations before it complete before operations after it begin. Currently a no-op in single-threaded mode; meaningful when the execution model is parallelized.

---

## Layer 4 — Runtime

The runtime implements LLaMA architecture on top of the instruction set. The key constraint for large models: the full model doesn't fit in RAM.

### Layer Streaming

Llama 3.2 1B weighs ~4.5GB dequantized. Available RAM: ~4GB. The solution is to never hold more than one layer's weights in memory at once.

```
Resident in memory (always):
  token_embedding  [128256, 2048]  ~1.0 GB
  rms_final        [2048]          tiny

Per forward pass (streamed from disk per layer):
  wq, wk, wv, wo   attention projections
  w1, w2, w3       FFN projections
  rms_att, rms_ffn  layer norms

KV-cache (persistent):
  16 layers × 2 × [2048, 256]  ~256 MB

Peak memory: ~1.8 GB for a 4.5 GB model.
```

When `cacheRawData: true` (default), raw Q4 bytes for all layers are pre-loaded into a single contiguous SharedArrayBuffer at startup. Subsequent token generation reads directly from RAM with no disk I/O — the bottleneck becomes compute, not I/O.

### Forward Pass (Single Token Decode)

```
for each layer (0..15):
  1.  RMS norm on residual stream
  2.  QKV projections (Q4 matvec × 3)
  3.  RoPE position encoding on Q, K
  4.  Update KV-cache at current position
  5.  Multi-head attention (scores → softmax → weighted sum)
  6.  Output projection (Q4 matvec)
  7.  Add to residual stream
  8.  RMS norm on residual
  9.  SwiGLU FFN: gate = silu(W1·x) * W3·x
  10. Down projection (Q4 matvec)
  11. Add to residual stream

LM head:
  12. Final RMS norm
  13. Project to vocab logits [128256] (128K dot products)
  14. Sample from top-K with temperature
```

### Grouped Query Attention

Llama 3.2 1B uses GQA: 32 query heads, 8 KV heads. Each KV head is shared by 4 query heads. The KV-cache stores 8 heads, not 32 — a 4× memory reduction.

```js
const headsPerKvHead = nHeads / nKvHeads;       // 32 / 8 = 4
const kvH = Math.floor(h / headsPerKvHead);     // which KV head for query h
```

---

## Quantization

### Q4_0

The default format for Llama 3.2 1B.

```
Block of 32 values:
  [2 bytes: float16 scale][16 bytes: 32 × 4-bit weights]

Nibble packing (NOT interleaved):
  byte[0]  = values[0]  (low 4 bits) | values[16] (high 4 bits)
  byte[1]  = values[1]  (low 4 bits) | values[17] (high 4 bits)
  ...
  byte[15] = values[15] (low 4 bits) | values[31] (high 4 bits)

Dequantize: float_val = (nibble - 8) * scale
```

18 bytes per block of 32 values = 4.5 bits/weight average.

### Q4_1

Same layout but adds a per-block minimum value:

```
[2 bytes: float16 delta][2 bytes: float16 min][16 bytes: nibbles]

Dequantize: float_val = nibble * delta + min
```

20 bytes per block.

### Why not dequantize before compute?

Dequantizing 280MB of weights per forward pass takes longer than computing matvec directly on the nibbles. The Q4 SIMD kernel computes `dot(x, W)` directly from packed bytes, applying the scale per block. No intermediate float buffer. No transpose.

---

## Parallelism

### WASM SIMD (always on)

128-bit SIMD processes 4 float32 values simultaneously. The Q4 kernel processes 32 weight values per SIMD iteration. Throughput: ~12 GFLOP/s.

### Worker Threads (default on)

Large FFN matrices (8192 rows) are split across main thread and one worker:

```
Main thread:   rows [4096..8191]  →  bottom-half result
Worker thread: rows [0..4095]    →  top-half result (via postMessage)
                                     shared weights via SharedArrayBuffer
```

The worker receives a reference to the SharedArrayBuffer once at startup — the 730MB weight cache is shared without copying. Only the input vector (dim × 4 bytes = 8KB) is copied per operation via postMessage transfer.

Attention matrices (2048 rows) run single-threaded. At ~3ms per matrix, the ~1.5ms postMessage round-trip overhead exceeds the parallelism gain.

Speedup: 3.46 → 3.61 tok/sec (4%). The SIMD kernel already made individual matvecs fast enough that threading has limited headroom.

---

## GGUF Format

PureBee reads GGUF v2/v3 — the format used by llama.cpp and Hugging Face.

```
File layout:
  [4 bytes: magic 0x46554747]
  [4 bytes: version]
  [8 bytes: tensor count]
  [8 bytes: metadata KV count]
  [metadata KV pairs...]
  [tensor info entries...]
  [alignment padding]
  [tensor data, contiguous]

Tensor dims: [ne0=cols, ne1=rows] (column-major naming, row-major data)
```

The streaming loader reads only the header (first 64MB at most) at startup. Tensor data is mapped by file offset and read on demand. With `cacheRawData`, the entire data section is read once into SharedArrayBuffer — subsequent access is pure memory.

---

## How to Go Further

PureBee is a working foundation, not a finished product. Every layer is designed to be extended, replaced, or built on. Here's what that looks like in practice.

**New operation:** Add a method to `PureBee` (instruction set) that calls an `ExecutionEngine` method. The engine method can have a WASM fast path and a JS fallback.

**New model:** Implement a runtime class that calls PureBee instructions. The instruction set handles the compute; the runtime handles the architecture.

**New quantization type:** Add dequantization in `gguf.js loadTensorData()` and a WASM kernel in `wasm-q4.js`. The rest of the stack is unchanged.

**Browser deployment:** The engine uses standard WebAssembly and SharedArrayBuffer (requires COOP/COEP headers). The runtime uses Node.js `fs` for model loading — replace with `fetch` + streaming reads. No other platform-specific code exists.

---

*The computation is the thing, not the substrate.*
