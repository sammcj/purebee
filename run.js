/**
 * PureBee — Software-Defined GPU
 *
 * Runs real AI models (LLaMA transformers) in pure Node.js.
 * No hardware dependencies. No CUDA. No silicon. Just software.
 *
 * Architecture: L1 Memory → L2 Engine → L3 Instructions → L4 LLaMA Runtime
 * Optimizations: Cache-friendly tiled matmul, Q8_0 quantization
 *
 * Usage:
 *   node run.js                        — run stories15M
 *   node run.js 42M                    — run stories42M
 *   node run.js 110M --q8              — run 110M quantized
 *   node run.js 42M --tokens=200       — generate 200 tokens
 *   node run.js --prompt="The cat"     — custom prompt
 *
 * Download models first: node download.js [15M|42M|110M|all]
 * Zero external dependencies.
 */

'use strict';

const path = require('path');
const fs = require('fs');
const { loadKarpathyModel } = require('./model-loader');
const { BPETokenizer } = require('./bpe-tokenizer');
const { LlamaRuntime, LlamaConfig } = require('./llama');
const { quantizeWeights } = require('./quantize');

// ── ANSI colors ──
const C = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  cyan: '\x1b[36m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  dim: '\x1b[2m',
  bold: '\x1b[1m',
  accent: '\x1b[38;5;48m',
  magenta: '\x1b[35m',
};

function log(msg) { console.log(msg); }
function header(msg) { log(`\n${C.bold}${C.accent}${msg}${C.reset}`); }
function step(msg) { log(`${C.cyan}▸${C.reset} ${msg}`); }
function success(msg) { log(`${C.green}✓${C.reset} ${msg}`); }
function info(msg) { log(`  ${C.dim}${msg}${C.reset}`); }

// ── Parse CLI args ──
function parseArgs() {
  const args = process.argv.slice(2);
  let modelSize = '15M';
  let quantize = false;
  let maxTokens = 100;
  let prompt = null;

  for (const arg of args) {
    if (arg === '--q8' || arg === '--quantize') {
      quantize = true;
    } else if (arg.startsWith('--tokens=')) {
      maxTokens = parseInt(arg.split('=')[1], 10);
    } else if (arg.startsWith('--prompt=')) {
      prompt = arg.split('=').slice(1).join('=');
    } else if (['15M', '42M', '110M'].includes(arg.toUpperCase())) {
      modelSize = arg.toUpperCase();
    }
  }

  return { modelSize, quantize, maxTokens, prompt };
}

function banner(modelSize, quantize) {
  const qStr = quantize ? ' + Q8' : '';
  const label = `stories${modelSize}${qStr}`;
  const pad = 39 - label.length;
  log(`${C.accent}${C.bold}`);
  log('  ╔═══════════════════════════════════════════════╗');
  log('  ║   PureBee — A GPU defined entirely in software                 ║');
  log(`  ║   ${label}${' '.repeat(pad > 0 ? pad : 0)}║`);
  log('  ╚═══════════════════════════════════════════════╝');
  log(C.reset);
}

async function main() {
  const { modelSize, quantize, maxTokens, prompt: customPrompt } = parseArgs();
  banner(modelSize, quantize);

  const modelsDir = path.join(__dirname, 'models');
  const modelFile = `stories${modelSize}.bin`;
  const modelPath = path.join(modelsDir, modelFile);
  const tokenizerPath = path.join(modelsDir, 'tokenizer.bin');

  // ── CHECK FILES EXIST ──
  if (!fs.existsSync(modelPath)) {
    log(`${C.red}  Model file not found: ${modelFile}${C.reset}`);
    log(`  Run: node download.js ${modelSize}`);
    log('');
    process.exit(1);
  }
  if (!fs.existsSync(tokenizerPath)) {
    log(`${C.red}  Tokenizer not found. Run: node download.js${C.reset}`);
    process.exit(1);
  }

  // ── STEP 1: Load Model Weights ──
  header('STEP 1 — Loading Model Weights');
  step(`Parsing ${modelFile}...`);

  const loadStart = Date.now();
  const { config: modelConfig, weights, sharedWeights } = loadKarpathyModel(modelPath);
  const loadTime = Date.now() - loadStart;

  success(`Model loaded in ${loadTime}ms`);
  info(`Architecture: LLaMA (${modelConfig.nLayers} layers, dim=${modelConfig.dim}, ${modelConfig.nHeads} heads)`);
  info(`Vocabulary: ${modelConfig.vocabSize} tokens, max seq: ${modelConfig.seqLen}`);

  // Count parameters
  let totalParams = 0;
  for (const data of Object.values(weights)) {
    if (data instanceof Float32Array) totalParams += data.length;
  }
  info(`Parameters: ${(totalParams / 1e6).toFixed(1)}M`);

  // ── STEP 2: Quantize (optional) ──
  let activeWeights = weights;
  let quantStats = null;

  if (quantize) {
    header('STEP 2 — Q8_0 Quantization');
    step('Quantizing weight matrices...');

    const qStart = Date.now();

    // Build shape map for quantizeWeights
    const { dim, hiddenDim, nLayers, nKvHeads, headDim } = modelConfig;
    const kvDim = nKvHeads * headDim;
    const shapes = {};
    for (let l = 0; l < nLayers; l++) {
      shapes[`layer${l}.wq`] = [dim, dim];
      shapes[`layer${l}.wk`] = [dim, kvDim];
      shapes[`layer${l}.wv`] = [dim, kvDim];
      shapes[`layer${l}.wo`] = [dim, dim];
      shapes[`layer${l}.w1`] = [dim, hiddenDim];
      shapes[`layer${l}.w2`] = [hiddenDim, dim];
      shapes[`layer${l}.w3`] = [dim, hiddenDim];
    }

    const result = quantizeWeights(weights, shapes, 'q8_0');
    activeWeights = result.weights;
    quantStats = result;

    const qTime = Date.now() - qStart;
    success(`Quantized in ${qTime}ms`);
    info(`Float32: ${result.originalMB}MB → Q8_0: ${result.quantizedMB}MB (${result.ratio}x compression)`);
  } else {
    step('Running in float32 mode (use --q8 for quantized)');
  }

  // ── STEP 3: Load Tokenizer ──
  header(quantize ? 'STEP 3 — Loading BPE Tokenizer' : 'STEP 2 — Loading BPE Tokenizer');
  step('Parsing tokenizer vocabulary...');

  const tokenizer = new BPETokenizer();
  tokenizer.load(tokenizerPath, modelConfig.vocabSize);

  const testStr = 'Once upon a time';
  const testEnc = tokenizer.encode(testStr);
  const testDec = tokenizer.decode(testEnc);
  success(`Tokenizer: "${testStr}" → [${testEnc.length} tokens] → "${testDec}"`);

  // ── STEP 4: Initialize Runtime ──
  header(quantize ? 'STEP 4 — Initializing LLaMA Runtime' : 'STEP 3 — Initializing LLaMA Runtime');
  step('Loading weights into PureBee memory...');

  const llamaConfig = new LlamaConfig({
    dim:       modelConfig.dim,
    hiddenDim: modelConfig.hiddenDim,
    nLayers:   modelConfig.nLayers,
    nHeads:    modelConfig.nHeads,
    nKvHeads:  modelConfig.nKvHeads,
    vocabSize: modelConfig.vocabSize,
    seqLen:    modelConfig.seqLen,
    headDim:   modelConfig.headDim,
  });

  const llama = new LlamaRuntime(llamaConfig, {
    log: false,
  });
  llama.loadWeights(activeWeights, sharedWeights);

  success('PureBee runtime initialized');
  const stats = llama.gpu.stats();
  info(`${stats.memory.tensors} tensors, ${stats.memory.totalMB}MB in PureBee memory`);

  // ── STEP 5: Verification Forward Pass ──
  const stepN = quantize ? 'STEP 5' : 'STEP 4';
  header(`${stepN} — Verification Forward Pass`);
  step('Running single forward pass to verify architecture...');

  const verifyTokens = tokenizer.encode('Once');
  const verifyStart = Date.now();
  const verifyLogits = llama.forward(verifyTokens, 0);
  const verifyTime = Date.now() - verifyStart;

  const indexed = Array.from(verifyLogits).map((v, i) => [v, i]);
  indexed.sort((a, b) => b[0] - a[0]);
  const top5 = indexed.slice(0, 5).map(([v, i]) => `"${tokenizer.vocab[i]}" (${v.toFixed(2)})`);

  success(`Forward pass completed in ${verifyTime}ms`);
  info(`Top 5 predictions: ${top5.join(', ')}`);

  // ── STEP 6: Text Generation ──
  const genStep = quantize ? 'STEP 6' : 'STEP 5';
  header(`${genStep} — Text Generation`);

  const prompts = customPrompt
    ? [customPrompt]
    : ['Once upon a time', 'The little dog', 'A brave knight'];

  const allResults = [];

  for (const promptText of prompts) {
    log('');
    step(`Prompt: "${promptText}"`);
    info(`Generating ${maxTokens} tokens...`);
    log('');

    const promptTokens = [tokenizer.bosId, ...tokenizer.encode(promptText)];
    let prevToken = tokenizer.bosId;

    process.stdout.write(`  ${C.cyan}${promptText}${C.reset}`);

    const result = llama.generate(promptTokens, maxTokens, {
      temperature: 0.8,
      topK: 40,
      eosId: tokenizer.eosId,
      onToken: (tokenId) => {
        const tokenStr = tokenizer.decodeToken(tokenId, prevToken);
        process.stdout.write(`${C.yellow}${tokenStr}${C.reset}`);
        prevToken = tokenId;
      },
    });

    allResults.push(result);

    log('');
    log('');
    info(`Prefill: ${result.prefillTime}ms | Decode: ${result.decodeTime}ms | ${C.reset}${C.bold}${result.tokPerSec.toFixed(1)} tok/sec${C.reset}${C.dim} | ${result.generated} tokens`);
  }

  // ── FINAL STATS ──
  header('PureBee — SYSTEM STATS');
  const finalStats = llama.gpu.stats();
  const avgTokSec = allResults.reduce((sum, r) => sum + r.tokPerSec, 0) / allResults.length;

  log(`  ${C.accent}Model${C.reset}       stories${modelSize} — ${modelConfig.nLayers} layers, dim=${modelConfig.dim}, ${(totalParams / 1e6).toFixed(1)}M params`);
  log(`  ${C.accent}Precision${C.reset}   ${quantize ? `Q8_0 (${quantStats.originalMB}MB → ${quantStats.quantizedMB}MB, ${quantStats.ratio}x compression)` : 'float32'}`);
  log(`  ${C.accent}Execution${C.reset}   Single-threaded`);
  log(`  ${C.accent}Speed${C.reset}       ${avgTokSec.toFixed(1)} tok/sec average`);
  log(`  ${C.accent}Memory${C.reset}      ${finalStats.memory.totalMB}MB across ${finalStats.memory.tensors} tensors`);
  log(`  ${C.accent}Operations${C.reset}  ${finalStats.ops} PureBee instructions dispatched`);
  log(`  ${C.accent}Engine${C.reset}      ${(finalStats.engine.flops / 1e6).toFixed(1)}M FLOPs executed`);
  log(`  ${C.accent}Tokenizer${C.reset}   SentencePiece BPE — ${modelConfig.vocabSize} tokens`);
  log(`  ${C.accent}Runtime${C.reset}     Pure Node.js — zero dependencies`);
  log(`  ${C.accent}Substrate${C.reset}   Software-defined — no GPU, no CUDA, no silicon`);
  log('');
  log(`${C.green}${C.bold}  A GPU defined as a specification, not hardware.${C.reset}`);
  log(`${C.dim}  Pure software. No GPU. No CUDA. No silicon.${C.reset}`);
  log('');

  // Shutdown workers
  llama.shutdown();
}

main().catch(err => {
  console.error(`\n${C.red}Error: ${err.message}${C.reset}`);
  console.error(err.stack);
  process.exit(1);
});
