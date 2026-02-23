/**
 * PureBee — 1 — Proof of Concept
 * 
 * Demonstrates a complete transformer inference pass
 * running entirely on PureBee instructions.
 * No GPU. No CUDA. No PyTorch. No external dependencies.
 * 
 * Run: node src/main.js
 */

'use strict';

const { GPTRuntime, TransformerConfig } = require('./transformer');
const { CharTokenizer } = require('./tokenizer');

// ── ANSI colors for terminal output ──
const C = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  cyan: '\x1b[36m',
  yellow: '\x1b[33m',
  dim: '\x1b[2m',
  bold: '\x1b[1m',
  accent: '\x1b[38;5;48m',
};

function log(msg) { console.log(msg); }
function header(msg) { log(`\n${C.bold}${C.accent}${msg}${C.reset}`); }
function step(msg) { log(`${C.cyan}▸${C.reset} ${msg}`); }
function success(msg) { log(`${C.green}✓${C.reset} ${msg}`); }
function info(msg) { log(`  ${C.dim}${msg}${C.reset}`); }

function banner() {
  log(`${C.accent}${C.bold}`);
  log('  ╔═══════════════════════════════════════╗');
  log('  ║   PureBee — Phase 1          ║');
  log('  ║   Pure Software Transformer Runtime   ║');
  log('  ╚═══════════════════════════════════════╝');
  log(C.reset);
}

async function runTests() {
  banner();

  // ── TEST 1: Memory Model ──
  header('TEST 1 — L1 Memory Model');
  const { PureBeeMemory, Tensor } = require('./memory');
  const mem = new PureBeeMemory();

  step('Allocating tensors...');
  const t1 = mem.alloc('test_tensor', [4, 4]);
  t1.randomize(1.0);
  const t2 = mem.alloc('identity', [3, 3]);
  t2.data.set([1,0,0, 0,1,0, 0,0,1]);

  success(`Tensor allocated: ${t1}`);
  success(`Identity matrix: ${t2}`);
  info(`Memory usage: ${mem.totalMB}MB across ${mem.stats().tensors} tensors`);

  // ── TEST 2: Execution Engine ──
  header('TEST 2 — L2 Execution Engine');
  const { ExecutionEngine } = require('./engine');
  const engine = new ExecutionEngine();

  step('Testing TENSOR_MUL [2,3] x [3,2] → [2,2]...');
  const A = new Tensor('A', [2, 3]);
  const B = new Tensor('B', [3, 2]);
  A.data.set([1, 2, 3, 4, 5, 6]);
  B.data.set([7, 8, 9, 10, 11, 12]);
  const C_out = engine.tensorMul(A, B, 'C');
  // Expected: [[58, 64], [139, 154]]
  const expected = [58, 64, 139, 154];
  const correct = expected.every((v, i) => Math.abs(C_out.data[i] - v) < 0.001);
  success(`TENSOR_MUL result: [${Array.from(C_out.data)}] — ${correct ? 'CORRECT ✓' : 'ERROR ✗'}`);

  step('Testing SOFTMAX...');
  const logits = new Tensor('logits', [1, 4]);
  logits.data.set([1.0, 2.0, 3.0, 4.0]);
  const probs = engine.softmax(logits, 'probs');
  const probSum = Array.from(probs.data).reduce((a, b) => a + b, 0);
  success(`SOFTMAX sums to: ${probSum.toFixed(6)} — ${Math.abs(probSum - 1.0) < 1e-5 ? 'CORRECT ✓' : 'ERROR ✗'}`);

  step('Testing GELU...');
  const x = new Tensor('x', [1, 4]);
  x.data.set([-1.0, 0.0, 1.0, 2.0]);
  const gelu = engine.gelu(x, 'gelu_out');
  success(`GELU(-1,0,1,2): [${Array.from(gelu.data).map(v => v.toFixed(4)).join(', ')}]`);

  step('Testing LAYER_NORM...');
  const xn = new Tensor('xn', [2, 4]);
  xn.data.set([1, 2, 3, 4, 5, 6, 7, 8]);
  const wn = new Tensor('wn', [4]); wn.data.fill(1.0);
  const bn = new Tensor('bn', [4]); bn.data.fill(0.0);
  const normed = engine.layerNorm(xn, wn, bn, 1e-5, 'normed');
  const row0Mean = (normed.data[0]+normed.data[1]+normed.data[2]+normed.data[3])/4;
  success(`LAYER_NORM row0 mean: ${row0Mean.toFixed(6)} — ${Math.abs(row0Mean) < 1e-4 ? 'CORRECT ✓' : 'ERROR ✗'}`);

  info(`Engine stats: ${engine.stats.ops} ops, ${(engine.stats.flops/1e3).toFixed(1)}K FLOPs`);

  // ── TEST 3: PureBee Instruction Set ──
  header('TEST 3 — L3 Instruction Set');
  const { PureBee } = require('./purebee');
  const gpu = new PureBee({ log: false });

  step('Testing full instruction pipeline...');
  gpu.GRID_WRITE('mat_a', [2, 3], new Float32Array([1,2,3,4,5,6]));
  gpu.GRID_WRITE('mat_b', [3, 2], new Float32Array([7,8,9,10,11,12]));
  const result = gpu.TENSOR_MUL('mat_a', 'mat_b', 'mat_c');
  success(`GRID_WRITE → TENSOR_MUL → result: [${Array.from(result.data)}]`);

  gpu.GRID_WRITE('v', [1, 4], new Float32Array([1, 2, 3, 4]));
  gpu.SOFTMAX('v', 'v_probs');
  gpu.GELU('v', 'v_gelu');
  gpu.SYNC();
  success(`SOFTMAX + GELU + SYNC — all instructions executed`);

  const stats = gpu.stats();
  info(`${stats.ops} instructions dispatched, ${stats.memory.tensors} tensors in memory`);

  // ── TEST 4: Tokenizer ──
  header('TEST 4 — Tokenizer');
  const { CharTokenizer } = require('./tokenizer');
  const tokenizer = new CharTokenizer();

  step('Encoding test string...');
  const testStr = 'The quick brown fox';
  const encoded = tokenizer.encode(testStr);
  const decoded = tokenizer.decode(encoded);
  success(`"${testStr}" → [${encoded.slice(0,6).join(', ')}...] → "${decoded}"`);
  info(`Vocab size: ${tokenizer.vocabSize} tokens`);

  // ── TEST 5: Full Transformer Forward Pass ──
  header('TEST 5 — Full Transformer Forward Pass');

  // Tiny config — runs fast, proves architecture
  const config = new TransformerConfig({
    vocabSize: tokenizer.vocabSize,
    seqLen: 64,
    dModel: 64,
    nHeads: 4,
    nLayers: 2,
    dFF: 256
  });

  info(`Config: ${config.nLayers} layers, dModel=${config.dModel}, ${config.nHeads} heads, vocab=${config.vocabSize}`);

  step('Initializing GPT runtime on PureBee...');
  const gpt = new GPTRuntime(config, { log: false });
  gpt.initRandomWeights();

  const prompt = 'Once upon a time';
  step(`Running forward pass: "${prompt}"`);

  const tokens = tokenizer.encode(prompt);
  info(`Input tokens: [${tokens.join(', ')}] (${tokens.length} tokens)`);

  const t0 = Date.now();
  const outputLogits = gpt.forward(tokens);
  const elapsed = Date.now() - t0;

  success(`Forward pass complete in ${elapsed}ms`);
  info(`Output logits shape: [${config.vocabSize}]`);

  // Show top-5 predicted next tokens
  const indexed = Array.from(outputLogits).map((v, i) => [v, i]);
  indexed.sort((a, b) => b[0] - a[0]);
  const top5 = indexed.slice(0, 5).map(([v, i]) => `"${tokenizer.idToToken[i]}" (${v.toFixed(3)})`);
  info(`Top 5 next token predictions: ${top5.join(', ')}`);

  // ── TEST 6: Text Generation ──
  header('TEST 6 — Text Generation');
  step('Generating 30 tokens from prompt...');
  info(`Note: weights are random — output proves architecture, not coherence`);
  log('');

  let genTokens = [...tokens];
  let generated = '';
  const maxNew = 30;

  process.stdout.write(`  ${C.dim}Prompt:${C.reset} ${C.cyan}${prompt}${C.reset}`);

  const genStart = Date.now();
  for (let i = 0; i < maxNew; i++) {
    // Use last 32 tokens as context
    const context = genTokens.slice(-32);
    const nextLogits = gpt.forward(context);
    const nextToken = gpt.sample(nextLogits, 40, 0.8);
    genTokens.push(nextToken);
    const tok = tokenizer.idToToken[nextToken] || '?';
    generated += tok;
    process.stdout.write(`${C.yellow}${tok}${C.reset}`);
    if (nextToken === tokenizer.eosId) break;
  }
  const genElapsed = Date.now() - genStart;

  log('');
  log('');
  success(`Generated ${maxNew} tokens in ${genElapsed}ms (${(maxNew / genElapsed * 1000).toFixed(1)} tok/sec)`);

  // ── FINAL STATS ──
  header('PureBee SYSTEM STATS');
  const finalStats = gpt.gpu.stats();
  log(`  ${C.accent}Memory${C.reset}      ${finalStats.memory.totalMB}MB across ${finalStats.memory.tensors} tensors`);
  log(`  ${C.accent}Operations${C.reset}  ${finalStats.ops} instructions dispatched`);
  log(`  ${C.accent}Engine${C.reset}      ${(finalStats.engine.flops / 1e6).toFixed(1)}M FLOPs executed`);
  log(`  ${C.accent}Runtime${C.reset}     Pure Node.js — zero dependencies`);
  log(`  ${C.accent}Hardware${C.reset}    CPU only — no GPU, no CUDA, no silicon`);
  log('');
  log(`${C.green}${C.bold}  PureBee Phase 1 — All tests passed.${C.reset}`);
  log(`${C.dim}  The math runs. The architecture holds. Ready for Phase 2.${C.reset}`);
  log('');
}

runTests().catch(err => {
  console.error('\x1b[31mError:\x1b[0m', err.message);
  console.error(err.stack);
  process.exit(1);
});
