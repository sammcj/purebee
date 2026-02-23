# Contributing

PureBee is a working foundation. Contributions that are clear, auditable, and principled are welcome.

## Before You Write Code

Open an issue first. Describe what you want to change and why. This keeps effort from being wasted and lets us agree on direction before anyone writes a line.

## What Good Contributions Look Like

PureBee has four clean layers with clear boundaries. Good contributions respect those boundaries — the instruction set doesn't know what SIMD is, the engine doesn't know what a transformer is. If a change blurs a layer boundary, it needs a strong justification.

Specifically:
- **New operations** belong in `purebee.js` with an engine implementation in `engine.js`
- **New quantization formats** belong in `gguf.js` and `wasm-q4.js`
- **New models** belong in a new runtime file, using only existing PureBee instructions
- **Performance improvements** should include before/after tok/sec numbers

## Code Style

- No external dependencies. Zero is the number.
- No build step. If it requires a compiler or toolchain, it doesn't belong here.
- Every function should be readable without context. Comments explain *why*, not *what*.
- If you're adding a WASM kernel, the binary construction must remain in JavaScript — no precompiled blobs.

## Pull Requests

- One thing per PR. Small and focused beats large and ambitious.
- Include a description of what changed and why.
- Run the smoke test before submitting:

```bash
node --max-old-space-size=4096 -e "
const { parseHeader, StreamingWeightLoader } = require('./streaming-loader');
const { StreamingLlamaRuntime } = require('./llama-streaming');
const path = require('path');
async function run() {
  const modelPath = path.join(__dirname, 'models', 'Llama-3.2-1B-Instruct-Q4_0.gguf');
  const { config, tokenizer, tensorIndex, sharedWeights } = parseHeader(modelPath);
  config.seqLen = 2048;
  const loader = new StreamingWeightLoader(modelPath, tensorIndex, { cacheRawData: true });
  const rt = new StreamingLlamaRuntime(config, { earlyExit: false });
  rt.loader = loader; rt._sharedWeights = sharedWeights;
  rt.loadResidentWeights(loader);
  const prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n';
  const tokens = tokenizer.encode(prompt);
  const result = rt.generate(tokens, 8, { temperature: 0.1, topK: 1, greedy: true, eosId: -1 });
  const text = result.tokens.slice(tokens.length).map(t => tokenizer.decode([t])).join('');
  console.log('Output:', text);
  console.log('Speed:', result.tokPerSec.toFixed(2), 'tok/sec');
  rt.shutdown(); process.exit(0);
}
run().catch(e => { console.error(e); process.exit(1); });
"
```

Must output `2 + 2 = 4`. If it doesn't, the PR won't merge.

## Licensing

By submitting a pull request, you agree to license your contribution under the same [FSL-1.1](./LICENSE) terms as the project. This allows PureBee to offer commercial licenses without fragmenting the codebase.

## Questions

Open an issue or reach out at [license@purebee.io](mailto:license@purebee.io).
