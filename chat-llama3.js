/**
 * PureBee — 6 — Llama 3.2 Chat Interface
 *
 * Interactive chat with Llama 3.2 1B using layer streaming.
 * The model is too large to fit in memory — weights are loaded
 * one layer at a time from disk during each forward pass.
 *
 * Features:
 *   - Layer streaming: ~1.8GB peak memory for a 4.5GB model
 *   - Early exit: skip layers when prediction confidence is high
 *   - Speculative decoding: draft with early exit, verify with full model
 *   - WASM SIMD acceleration for matmul
 *
 * Usage:
 *   node --max-old-space-size=4096 chat-llama3.js
 *   node --max-old-space-size=4096 chat-llama3.js --no-speculative
 *   node --max-old-space-size=4096 chat-llama3.js --no-early-exit
 *
 * Commands:
 *   /reset    — clear conversation context
 *   /temp N   — set temperature
 *   /tokens N — set max tokens per response
 *   /stats    — show runtime statistics
 *   /quit     — exit
 *
 * Download model: node download.js llama3
 * Zero external dependencies.
 */

'use strict';

const path = require('path');
const fs = require('fs');
const readline = require('readline');
const { parseHeader, StreamingWeightLoader } = require('./streaming-loader');
const { StreamingLlamaRuntime } = require('./llama-streaming');

// ── ANSI ──
const C = {
  reset: '\x1b[0m', green: '\x1b[32m', cyan: '\x1b[36m',
  yellow: '\x1b[33m', red: '\x1b[31m', dim: '\x1b[2m',
  bold: '\x1b[1m', accent: '\x1b[38;5;48m', magenta: '\x1b[35m',
};

function log(msg = '') { console.log(msg); }

// ═══════════════════════════════════════════════════════════════════════════
// Llama 3 Chat Template
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Build a Llama 3 chat prompt.
 *
 * Format:
 *   <|begin_of_text|><|start_header_id|>system<|end_header_id|>
 *   {system_message}<|eot_id|>
 *   <|start_header_id|>user<|end_header_id|>
 *   {user_message}<|eot_id|>
 *   <|start_header_id|>assistant<|end_header_id|>
 */
function buildLlama3Prompt(userInput, history) {
  const parts = [];

  // System prompt
  parts.push('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Answer concisely and clearly.<|eot_id|>');

  // Conversation history
  for (const turn of history) {
    if (turn.role === 'user') {
      parts.push(`<|start_header_id|>user<|end_header_id|>\n\n${turn.text}<|eot_id|>`);
    } else {
      parts.push(`<|start_header_id|>assistant<|end_header_id|>\n\n${turn.text}<|eot_id|>`);
    }
  }

  // Current user message
  parts.push(`<|start_header_id|>user<|end_header_id|>\n\n${userInput}<|eot_id|>`);

  // Start assistant response
  parts.push('<|start_header_id|>assistant<|end_header_id|>\n\n');

  return parts.join('');
}

// ═══════════════════════════════════════════════════════════════════════════
// Argument Parsing
// ═══════════════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    earlyExit: false,
    speculative: false,
    earlyExitThreshold: 0.9,
    earlyExitInterval: 4,
    noWasm: false,
    cacheRawData: true,
    threads: true,
  };

  for (const arg of args) {
    if (arg === '--no-early-exit') opts.earlyExit = false;
    if (arg === '--no-speculative') opts.speculative = false;
    if (arg === '--no-wasm') opts.noWasm = true;
    if (arg === '--no-cache') opts.cacheRawData = false;
    if (arg === '--no-threads') opts.threads = false;
    if (arg.startsWith('--threshold=')) opts.earlyExitThreshold = parseFloat(arg.split('=')[1]);
  }

  return opts;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const opts = parseArgs();

  // ── Banner ──
  log(`${C.accent}${C.bold}`);
  log('  ╔═══════════════════════════════════════════════╗');
  log('  ║   PureBee — Llama 3.2 1B Instruct           ║');
  log('  ║   Layer Streaming + WASM SIMD + Q4          ║');
  log('  ╚═══════════════════════════════════════════════╝');
  log(C.reset);

  // ── Check model file ──
  const modelsDir = path.join(__dirname, 'models');
  const modelPath = path.join(modelsDir, 'Llama-3.2-1B-Instruct-Q4_0.gguf');

  if (!fs.existsSync(modelPath)) {
    log(`${C.red}  Model not found: Llama-3.2-1B-Instruct-Q4_0.gguf${C.reset}`);
    log(`  Run: node download.js llama3`);
    process.exit(1);
  }

  // ── Check heap size ──
  const heapLimit = require('v8').getHeapStatistics().heap_size_limit;
  const heapGB = (heapLimit / (1024 * 1024 * 1024)).toFixed(1);
  if (heapLimit < 3 * 1024 * 1024 * 1024) {
    log(`${C.yellow}  Warning: Heap limit is ${heapGB}GB. Recommended: 4GB${C.reset}`);
    log(`${C.yellow}  Run with: node --max-old-space-size=4096 chat-llama3.js${C.reset}`);
    log();
  }

  // ── Parse GGUF header (fast — no weight loading) ──
  log(`${C.dim}  Parsing GGUF header...${C.reset}`);
  const headerStart = Date.now();
  const { config, tokenizer, tensorIndex, sharedWeights } = parseHeader(modelPath);

  // Cap context length — full 128K context would require too much KV cache memory
  const maxSeqLen = 2048;
  if (config.seqLen > maxSeqLen) {
    config.seqLen = maxSeqLen;
  }

  log(`${C.dim}  Header parsed in ${Date.now() - headerStart}ms${C.reset}`);
  log(`${C.dim}  Config: dim=${config.dim}, hidden=${config.hiddenDim}, layers=${config.nLayers}, heads=${config.nHeads}, kv_heads=${config.nKvHeads}${C.reset}`);
  log(`${C.dim}  Vocab: ${config.vocabSize} tokens, RoPE theta=${config.ropeTheta}, ctx=${config.seqLen}${C.reset}`);
  log(`${C.dim}  Tensor index: ${tensorIndex.size} tensors${C.reset}`);

  // ── Create streaming loader ──
  const loader = new StreamingWeightLoader(modelPath, tensorIndex, {
    cacheRawData: opts.cacheRawData,
  });

  // ── Create streaming runtime ──
  const llama = new StreamingLlamaRuntime(config, {
    loader,
    sharedWeights,
    noWasm: opts.noWasm,
    earlyExitThreshold: opts.earlyExit ? opts.earlyExitThreshold : 0,
    earlyExitInterval: opts.earlyExitInterval,
  });

  // ── Load resident weights (embedding, norm, lm_head) ──
  log(`${C.dim}  Loading resident weights (embedding + norms)...${C.reset}`);
  const residentStart = Date.now();
  llama.loadResidentWeights();
  log(`${C.dim}  Resident weights loaded in ${Date.now() - residentStart}ms${C.reset}`);

  // ── Thread pool (parallel matvec) ──
  let threaded = false;
  if (opts.threads && opts.cacheRawData) {
    threaded = await llama.initThreadPool();
  }

  // ── Find special tokens ──
  // Llama 3 uses these special tokens for chat:
  const specialTokenNames = [
    '<|begin_of_text|>', '<|end_of_text|>', '<|start_header_id|>',
    '<|end_header_id|>', '<|eot_id|>',
  ];
  const specialTokenIds = {};
  for (const name of specialTokenNames) {
    const id = tokenizer.getSpecialToken(name);
    if (id !== undefined) specialTokenIds[name] = id;
  }

  // Register any missing special tokens
  for (const name of specialTokenNames) {
    if (specialTokenIds[name] === undefined) {
      // Try to find in vocab directly
      for (let i = 0; i < tokenizer.vocab.length; i++) {
        if (tokenizer.vocab[i] === name) {
          specialTokenIds[name] = i;
          tokenizer._specialTokens.set(name, i);
          break;
        }
      }
    }
  }

  log(`${C.dim}  Special tokens: ${JSON.stringify(specialTokenIds)}${C.reset}`);

  // EOS tokens for Llama 3
  const eosIds = [tokenizer.eosId];
  if (specialTokenIds['<|eot_id|>'] !== undefined) eosIds.push(specialTokenIds['<|eot_id|>']);
  if (specialTokenIds['<|end_of_text|>'] !== undefined) eosIds.push(specialTokenIds['<|end_of_text|>']);

  // ── Speculative decoder (optional) ──
  let specDecoder = null;
  if (opts.speculative) {
    try {
      const { SelfSpeculativeDecoder } = require('./speculative');
      specDecoder = new SelfSpeculativeDecoder(llama, {
        draftLayers: 4,
        draftCount: 4,
        temperature: 0.6,
        topK: 40,
      });
      log(`${C.dim}  Speculative decoding: enabled (draft=4 layers, K=4 tokens)${C.reset}`);
    } catch (e) {
      log(`${C.dim}  Speculative decoding: not available (${e.message})${C.reset}`);
    }
  }

  // ── Ready ──
  log();
  log(`${C.green}  Ready.${C.reset} ${C.dim}${config.nLayers} layers, dim=${config.dim}, heap=${heapGB}GB${C.reset}`);
  log(`${C.dim}  Features: layer-streaming${opts.cacheRawData ? ' + Q4-cached' : ''}${threaded ? ' + 2-thread' : ''}${opts.earlyExit ? ' + early-exit' : ''}${specDecoder ? ' + speculative' : ''} + WASM-SIMD${C.reset}`);
  log();
  log(`${C.dim}  Commands: /reset  /temp N  /tokens N  /stats  /quit${C.reset}`);
  log();

  // ── Chat state ──
  let temperature = 0.6;
  let topK = 40;
  let maxTokens = 200;
  const history = [];

  // ── REPL ──
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: `${C.cyan}You > ${C.reset}`,
  });

  rl.prompt();

  rl.on('line', async (line) => {
    const input = line.trim();
    if (!input) { rl.prompt(); return; }

    // ── Commands ──
    if (input === '/quit' || input === '/exit') {
      log(`\n${C.dim}  Goodbye.${C.reset}`);
      llama.shutdown();
      process.exit(0);
    }

    if (input === '/reset') {
      llama.resetCache();
      history.length = 0;
      log(`${C.dim}  Conversation reset.${C.reset}\n`);
      rl.prompt();
      return;
    }

    if (input === '/stats') {
      const s = llama.stats;
      log(`${C.dim}  Runtime stats:${C.reset}`);
      log(`${C.dim}    Total forwards: ${s.totalForwards}${C.reset}`);
      log(`${C.dim}    Avg layers/forward: ${s.avgLayers} / ${config.nLayers}${C.reset}`);
      log(`${C.dim}    Early exits: ${s.earlyExits}${C.reset}`);
      log(`${C.dim}    WASM SIMD: ${s.wasmReady ? 'active' : 'inactive'}${C.reset}`);
      if (s.vocabPruning) {
      }
      if (specDecoder) {
        const ss = specDecoder.stats;
        log(`${C.dim}    Speculative: ${ss.totalRounds} rounds, ${ss.acceptanceRate} acceptance${C.reset}`);
      }
      const mem = process.memoryUsage();
      log(`${C.dim}    Memory: RSS=${(mem.rss / 1024 / 1024).toFixed(0)}MB, Heap=${(mem.heapUsed / 1024 / 1024).toFixed(0)}/${(mem.heapTotal / 1024 / 1024).toFixed(0)}MB${C.reset}`);
      log();
      rl.prompt();
      return;
    }

    if (input.startsWith('/temp')) {
      const val = parseFloat(input.split(/\s+/)[1]);
      if (!isNaN(val) && val > 0 && val <= 2) {
        temperature = val;
        log(`${C.dim}  Temperature set to ${temperature}${C.reset}\n`);
      } else {
        log(`${C.dim}  Usage: /temp 0.1-2.0 (current: ${temperature})${C.reset}\n`);
      }
      rl.prompt();
      return;
    }

    if (input.startsWith('/tokens')) {
      const val = parseInt(input.split(/\s+/)[1], 10);
      if (!isNaN(val) && val > 0 && val <= 1024) {
        maxTokens = val;
        log(`${C.dim}  Max tokens set to ${maxTokens}${C.reset}\n`);
      } else {
        log(`${C.dim}  Usage: /tokens 1-1024 (current: ${maxTokens})${C.reset}\n`);
      }
      rl.prompt();
      return;
    }

    // ── Build prompt ──
    const chatPrompt = buildLlama3Prompt(input, history);
    let promptTokens = tokenizer.encode(chatPrompt);

    // Truncate if too long
    const maxCtx = Math.floor(config.seqLen * 0.7);
    if (promptTokens.length > maxCtx) {
      // Keep system prompt (first ~50 tokens) + recent context
      promptTokens = promptTokens.slice(-maxCtx);
    }

    process.stdout.write(`\n${C.accent}Llama${C.reset} ${C.dim}>${C.reset} `);

    // ── Generate ──
    let prevToken = tokenizer.bosId;
    let responseText = '';
    const genStart = Date.now();

    if (specDecoder) {
      // Speculative generation
      const result = specDecoder.generate(promptTokens, maxTokens, {
        temperature,
        topK,
        eosIds,
        onToken: (tokenId) => {
          if (eosIds.includes(tokenId)) return;
          const text = tokenizer.decodeToken(tokenId, prevToken);
          responseText += text;
          process.stdout.write(`${C.yellow}${text}${C.reset}`);
          prevToken = tokenId;
        },
      });

      const elapsed = Date.now() - genStart;
      log();
      log(`${C.dim}  ${result.generated} tokens | ${result.tokPerSec.toFixed(1)} tok/s | ${elapsed}ms | accepted ${result.acceptanceRate}${C.reset}`);
    } else if (threaded) {
      // Threaded generation (async decode with parallel FFN matvec)
      const result = await llama.generateAsync(promptTokens, maxTokens, {
        temperature,
        topK,
        eosId: eosIds[0],
        onToken: (tokenId, idx, layersUsed) => {
          if (eosIds.includes(tokenId)) return;
          const text = tokenizer.decodeToken(tokenId, prevToken);
          responseText += text;
          process.stdout.write(`${C.yellow}${text}${C.reset}`);
          prevToken = tokenId;
        },
      });

      const elapsed = Date.now() - genStart;
      log();
      log(`${C.dim}  ${result.generated} tokens | ${result.tokPerSec.toFixed(1)} tok/s | ${elapsed}ms | avg ${result.avgLayers}/${config.nLayers} layers | early exits: ${result.earlyExits}${C.reset}`);
    } else {
      // Standard generation with layer streaming
      const result = llama.generate(promptTokens, maxTokens, {
        temperature,
        topK,
        eosId: eosIds[0],
        onToken: (tokenId, idx, layersUsed) => {
          if (eosIds.includes(tokenId)) return;
          const text = tokenizer.decodeToken(tokenId, prevToken);
          responseText += text;
          process.stdout.write(`${C.yellow}${text}${C.reset}`);
          prevToken = tokenId;
        },
      });

      const elapsed = Date.now() - genStart;
      log();
      log(`${C.dim}  ${result.generated} tokens | ${result.tokPerSec.toFixed(1)} tok/s | ${elapsed}ms | avg ${result.avgLayers}/${config.nLayers} layers | early exits: ${result.earlyExits}${C.reset}`);
    }

    log();

    // Update history
    history.push({ role: 'user', text: input });
    history.push({ role: 'assistant', text: responseText.trim() });

    // Keep history reasonable
    if (history.length > 8) history.splice(0, 2);

    // Reset cache between turns (stateless like chat.js)
    llama.resetCache();

    rl.prompt();
  });

  rl.on('close', () => {
    log(`\n${C.dim}  Goodbye.${C.reset}`);
    llama.shutdown();
    process.exit(0);
  });
}

main().catch(err => {
  console.error(`${C.red}Error: ${err.message}${C.reset}`);
  console.error(err.stack);
  process.exit(1);
});
