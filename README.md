# 🐝 PureBee

**A GPU defined entirely in software.**

No GPU. No CUDA. No hardware assumptions. No dependencies.

PureBee is pure math — four layers, zero dependencies, runs Llama 3.2 1B at **3.6 tok/sec on a single CPU core**.

---

## The Idea

A GPU is not a physical thing. It's a rule.

Thousands of cores doing simple math simultaneously on a grid of data. That's all it is. Strip away the silicon, the transistors, the electrons — and what remains is the math. A function, a grid, and a rule that says: *apply simultaneously*.

PureBee is proof of that claim.

If the math is the GPU, and electricity is just the mechanism that makes the math fast — then you don't need the electricity. You need the math.

PureBee replaces the hardware with a specification. The specification replaces the silicon with software. The software runs on anything.

---

## Architecture

Four layers. Clean boundaries. No hidden state.

```
┌─────────────────────────────────────┐
│           RUNTIME                   │  Execution, scheduling, output
├─────────────────────────────────────┤
│        INSTRUCTION SET              │  Operations: matmul, softmax, rope...
├─────────────────────────────────────┤
│            ENGINE                   │  Parallel rule application, SIMD
├─────────────────────────────────────┤
│            MEMORY                   │  Tensor layout, quantization, cache
└─────────────────────────────────────┘
```

Each layer is replaceable. Each layer is auditable. Each layer is just math.

---

## Performance

Starting from first principles, built layer by layer:

| Stage | Speed | What changed |
|-------|-------|-------------|
| Baseline | 0.08 tok/sec | Pure JavaScript, no optimization |
| Typed arrays | 0.21 tok/sec | Float32Array memory layout |
| WASM | 0.7 tok/sec | Compiled compute kernels |
| Q4 quantization | 1.3 tok/sec | 4-bit weight compression |
| SIMD | 3.0 tok/sec | 128-bit parallel ops |
| Worker threads | 3.6 tok/sec | Split large matrix rows |

**45× total speedup.** Single CPU core. No GPU.

This isn't a demo. This is Llama 3.2 1B running full inference.

---

## Models

| Model | Parameters | Use |
|-------|-----------|-----|
| Llama 3.2 1B | 1B | Primary — full capability |
| SmolLM2-135M | 135M | Fast, lightweight |
| stories15M | 15M | Development and testing |
| stories42M | 42M | Development and testing |

---

## Getting Started

Requires Node.js ≥ 20.

```bash
git clone https://github.com/PureBee/purebee
cd purebee
node download.js llama3                              # ~770MB
node --max-old-space-size=4096 chat-llama3.js
```

The heap flag is not optional — Llama 3.2 1B weights won't fit in Node's default heap.

No GPU required. No CUDA installation. No driver configuration.

If it can run Node.js ≥ 20, it can run PureBee.

---

## Why

Every major AI inference framework assumes hardware. CUDA assumes NVIDIA. Metal assumes Apple. Even the "portable" runtimes assume a GPU exists somewhere.

PureBee assumes nothing.

This matters because:

- **Accessibility** — AI inference should run on any device, not just ones with the right silicon
- **Transparency** — a spec you can read is a GPU you can understand
- **Portability** — if the spec runs, it runs everywhere
- **The principle** — the computation is the thing, not the substrate

The universe might be running physics on something we can't see. PureBee runs inference on something you can read.

---

## License

[FSL-1.1-Apache-2.0](./LICENSE) — Free for personal, research, and internal use. Converts to Apache 2.0 on 2028-02-21.

Using PureBee as an ingredient in your product is fine. Selling PureBee-as-a-service is not — that requires a commercial license.

Commercial / hosted use: [license@purebee.io](mailto:license@purebee.io)

---

## Status

Active development. Core inference is stable. Optimizations ongoing.

**Roadmap:**
- Browser deployment (SharedArrayBuffer + COOP/COEP headers)
- Additional model support
- Expanded instruction set
- PureBee spec v1.0 formal publication

---

## Contributing

PureBee is built on a simple idea executed carefully. Contributions that honor that — clear, auditable, principled — are welcome.

Open an issue. Start a discussion. Read the spec.

---

*PureBee — github.com/PureBee | purebee.io*

*The GPU was always math. We just removed the middleman.*
