/**
 * PureBee — Interactive Chat with Memory
 *
 * A conversational AI demo powered by PureBee. Supports two modes:
 *   - Story mode: Karpathy TinyStories models (story completion)
 *   - Chat mode: GGUF instruction-tuned models (real conversation)
 *
 * Chat mode includes persistent memory — the model remembers facts
 * from previous conversations via a local RAG-style memory file.
 *
 * Usage:
 *   node chat.js                — story mode with stories15M
 *   node chat.js 42M --q8      — story mode with 42M quantized
 *   node chat.js smollm        — chat mode with SmolLM2-135M-Instruct
 *
 * Commands:
 *   /reset    — clear conversation context
 *   /memory   — show stored memories
 *   /forget   — clear all memories
 *   /temp N   — set temperature
 *   /tokens N — set max tokens per response
 *   /quit     — exit
 *
 * Download models: node download.js [15M|42M|110M|smollm|all]
 * Zero external dependencies.
 */

'use strict';

const path = require('path');
const fs = require('fs');
const readline = require('readline');
const { loadKarpathyModel } = require('./model-loader');
const { BPETokenizer } = require('./bpe-tokenizer');
const { LlamaRuntime, LlamaConfig } = require('./llama');
const { quantizeWeights } = require('./quantize');

// ── ANSI ──
const C = {
  reset: '\x1b[0m', green: '\x1b[32m', cyan: '\x1b[36m',
  yellow: '\x1b[33m', red: '\x1b[31m', dim: '\x1b[2m',
  bold: '\x1b[1m', accent: '\x1b[38;5;48m', magenta: '\x1b[35m',
};

function log(msg = '') { console.log(msg); }

const MEMORY_FILE = path.join(__dirname, 'memory.json');

// ═══════════════════════════════════════════════════════════════════════════
// Conversation Memory (RAG-style)
// ═══════════════════════════════════════════════════════════════════════════

class ConversationMemory {
  constructor() {
    this.facts = [];      // Persistent facts: { text, keywords, timestamp }
    this.history = [];     // Current session conversation history
    this._load();
  }

  _load() {
    try {
      if (fs.existsSync(MEMORY_FILE)) {
        const data = JSON.parse(fs.readFileSync(MEMORY_FILE, 'utf-8'));
        this.facts = data.facts || [];
      }
    } catch { /* ignore corrupt file */ }
  }

  _save() {
    fs.writeFileSync(MEMORY_FILE, JSON.stringify({ facts: this.facts }, null, 2));
  }

  /** Add a user/assistant exchange to history. */
  addTurn(role, text) {
    this.history.push({ role, text, time: Date.now() });
  }

  /** Extract and store key facts from a conversation turn. */
  storeFact(text) {
    // Simple keyword extraction: keep sentences with names, numbers, or notable info
    const keywords = text.toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(w => w.length > 3);

    if (keywords.length > 0 && text.length > 10 && text.length < 500) {
      this.facts.push({
        text: text.trim(),
        keywords,
        timestamp: Date.now(),
      });
      // Keep only last 50 facts
      if (this.facts.length > 50) this.facts = this.facts.slice(-50);
      this._save();
    }
  }

  /** Retrieve relevant memories for a query using keyword matching. */
  retrieve(query, maxResults = 3) {
    if (this.facts.length === 0) return [];

    const queryWords = query.toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(w => w.length > 3);

    if (queryWords.length === 0) return this.facts.slice(-maxResults);

    // Score each fact by keyword overlap
    const scored = this.facts.map(fact => {
      let score = 0;
      for (const qw of queryWords) {
        for (const fw of fact.keywords) {
          if (fw.includes(qw) || qw.includes(fw)) score++;
        }
      }
      return { fact, score };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.filter(s => s.score > 0).slice(0, maxResults).map(s => s.fact);
  }

  /** Get recent conversation history as formatted text. */
  getRecentHistory(maxTurns = 4) {
    return this.history.slice(-maxTurns);
  }

  clearHistory() {
    this.history = [];
  }

  clearFacts() {
    this.facts = [];
    this._save();
  }

  get factCount() { return this.facts.length; }
}

// ═══════════════════════════════════════════════════════════════════════════
// Chat Template — Format prompts for instruction-tuned models
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Build a ChatML-formatted prompt for instruction-tuned models.
 * Includes system prompt, relevant memories, conversation history, and user input.
 */
function buildChatPrompt(userInput, memory, tokenizer) {
  const parts = [];

  // System prompt
  parts.push('<|im_start|>system');
  parts.push('You are a helpful, friendly assistant. Answer concisely.');

  // Inject relevant memories
  const relevantMemories = memory.retrieve(userInput, 2);
  if (relevantMemories.length > 0) {
    parts.push('Context from previous conversations:');
    for (const mem of relevantMemories) {
      parts.push(`- ${mem.text}`);
    }
  }
  parts.push('<|im_end|>');

  // Recent conversation history
  const history = memory.getRecentHistory(4);
  for (const turn of history) {
    if (turn.role === 'user') {
      parts.push(`<|im_start|>user`);
      parts.push(turn.text);
      parts.push('<|im_end|>');
    } else {
      parts.push(`<|im_start|>assistant`);
      parts.push(turn.text);
      parts.push('<|im_end|>');
    }
  }

  // Current user message
  parts.push('<|im_start|>user');
  parts.push(userInput);
  parts.push('<|im_end|>');

  // Start of assistant response
  parts.push('<|im_start|>assistant');

  return parts.join('\n');
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Loading — Supports both Karpathy and GGUF formats
// ═══════════════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  let modelId = '15M';
  let quantize = false;

  for (const arg of args) {
    if (arg === '--q8' || arg === '--quantize') quantize = true;
    else if (['15M', '42M', '110M'].includes(arg.toUpperCase())) modelId = arg.toUpperCase();
    else if (arg.toLowerCase() === 'smollm') modelId = 'smollm';
  }
  return { modelId, quantize };
}

function loadModel(modelId, quantize) {
  const modelsDir = path.join(__dirname, 'models');
  let config, weights, tokenizer, sharedWeights, isInstruct;

  if (modelId === 'smollm') {
    // ── GGUF model ──
    const ggufPath = path.join(modelsDir, 'SmolLM2-135M-Instruct-Q8_0.gguf');
    if (!fs.existsSync(ggufPath)) {
      log(`${C.red}  Model not found: SmolLM2-135M-Instruct-Q8_0.gguf${C.reset}`);
      log(`  Run: node download.js smollm`);
      process.exit(1);
    }

    const { loadGGUFModel } = require('./gguf');
    const result = loadGGUFModel(ggufPath);
    config = result.config;
    weights = result.weights;
    tokenizer = result.tokenizer;
    sharedWeights = result.sharedWeights;
    isInstruct = true;
  } else {
    // ── Karpathy model ──
    const modelPath = path.join(modelsDir, `stories${modelId}.bin`);
    const tokenizerPath = path.join(modelsDir, 'tokenizer.bin');

    if (!fs.existsSync(modelPath)) {
      log(`${C.red}  Model not found: stories${modelId}.bin${C.reset}`);
      log(`  Run: node download.js ${modelId}`);
      process.exit(1);
    }

    const result = loadKarpathyModel(modelPath);
    config = result.config;
    weights = result.weights;
    sharedWeights = result.sharedWeights;

    tokenizer = new BPETokenizer();
    tokenizer.load(tokenizerPath, config.vocabSize);
    isInstruct = false;

    // Optional quantization for Karpathy models
    if (quantize) {
      const { dim, hiddenDim, nLayers, nKvHeads, headDim } = config;
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
      const qResult = quantizeWeights(weights, shapes, 'q8_0');
      weights = qResult.weights;
      log(`${C.dim}  Quantized: ${qResult.originalMB}MB → ${qResult.quantizedMB}MB (${qResult.ratio}x)${C.reset}`);
    }
  }

  return { config, weights, tokenizer, sharedWeights, isInstruct };
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const { modelId, quantize } = parseArgs();

  // ── Banner ──
  const isGGUF = modelId === 'smollm';
  const modelLabel = isGGUF ? 'SmolLM2-135M-Instruct' : `stories${modelId}${quantize ? ' + Q8' : ''}`;
  const pad = 39 - modelLabel.length;
  log(`${C.accent}${C.bold}`);
  log('  ╔═══════════════════════════════════════════════╗');
  log('  ║   PureBee — Conversational AI               ║');
  log(`  ║   ${modelLabel}${' '.repeat(pad > 0 ? pad : 0)}║`);
  log('  ╚═══════════════════════════════════════════════╝');
  log(C.reset);

  // ── Load model ──
  log(`${C.dim}  Loading model...${C.reset}`);
  const { config, weights, tokenizer, sharedWeights, isInstruct } = loadModel(modelId, quantize);

  const llamaConfig = new LlamaConfig(config);
  const llama = new LlamaRuntime(llamaConfig, { log: false });
  llama.loadWeights(weights, sharedWeights);

  // Warm up
  llama.forward([tokenizer.bosId], 0);
  llama.resetCache();

  const stats = llama.gpu.stats();
  log(`${C.green}  Ready.${C.reset} ${C.dim}${config.nLayers} layers, dim=${config.dim}, ${stats.memory.totalMB}MB loaded${C.reset}`);

  // ── Memory ──
  const memory = new ConversationMemory();
  if (memory.factCount > 0) {
    log(`${C.dim}  Memory: ${memory.factCount} facts from previous sessions${C.reset}`);
  }

  log();
  if (isInstruct) {
    log(`${C.dim}  Chat with an instruction-tuned model. It understands questions.${C.reset}`);
    log(`${C.dim}  Memory persists across sessions (stored in memory.json).${C.reset}`);
  } else {
    log(`${C.dim}  Story completion mode — type the start of a story.${C.reset}`);
  }
  log(`${C.dim}  Commands: /reset  /memory  /forget  /temp N  /tokens N  /quit${C.reset}`);
  log();

  // ── Chat state ──
  let temperature = isInstruct ? 0.6 : 0.8;
  let topK = 40;
  let maxTokens = isInstruct ? 200 : 150;

  // Find EOS tokens for instruction models
  let eosIds = [tokenizer.eosId];
  if (isInstruct) {
    const imEnd = tokenizer.getSpecialToken('<|im_end|>');
    if (imEnd !== undefined) eosIds.push(imEnd);
    const eot = tokenizer.getSpecialToken('<|endoftext|>');
    if (eot !== undefined) eosIds.push(eot);
  }

  // ── REPL ──
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: `${C.cyan}You › ${C.reset}`,
  });

  rl.prompt();

  rl.on('line', (line) => {
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
      memory.clearHistory();
      log(`${C.dim}  Conversation reset.${C.reset}\n`);
      rl.prompt();
      return;
    }

    if (input === '/memory') {
      if (memory.factCount === 0) {
        log(`${C.dim}  No memories stored yet.${C.reset}\n`);
      } else {
        log(`${C.dim}  Stored memories (${memory.factCount}):${C.reset}`);
        for (const fact of memory.facts.slice(-10)) {
          log(`${C.dim}    - ${fact.text.slice(0, 80)}${fact.text.length > 80 ? '...' : ''}${C.reset}`);
        }
        log();
      }
      rl.prompt();
      return;
    }

    if (input === '/forget') {
      memory.clearFacts();
      memory.clearHistory();
      log(`${C.dim}  All memories cleared.${C.reset}\n`);
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
      if (!isNaN(val) && val > 0 && val <= config.seqLen) {
        maxTokens = val;
        log(`${C.dim}  Max tokens set to ${maxTokens}${C.reset}\n`);
      } else {
        log(`${C.dim}  Usage: /tokens 1-${config.seqLen} (current: ${maxTokens})${C.reset}\n`);
      }
      rl.prompt();
      return;
    }

    // ── Build prompt ──
    let promptTokens;

    if (isInstruct) {
      // ChatML format for instruction-tuned models
      // The template includes <|im_start|> which IS the BOS token, so don't double-add
      const chatPrompt = buildChatPrompt(input, memory, tokenizer);
      promptTokens = tokenizer.encode(chatPrompt);

      // Truncate if too long for context window (keep last portion)
      const maxCtx = Math.floor(config.seqLen * 0.7); // Leave room for generation
      if (promptTokens.length > maxCtx) {
        promptTokens = [tokenizer.bosId, ...promptTokens.slice(-maxCtx + 1)];
      }
    } else {
      // Simple story continuation
      promptTokens = [tokenizer.bosId, ...tokenizer.encode(input)];
    }

    process.stdout.write(`\n${C.accent}PureBee${C.reset} ${C.dim}›${C.reset} `);

    // ── Generate ──
    let prevToken = tokenizer.bosId;
    let responseText = '';

    const genStart = Date.now();
    const result = llama.generate(promptTokens, maxTokens, {
      temperature,
      topK,
      eosId: eosIds[0], // Primary EOS
      onToken: (tokenId) => {
        // Stop on any EOS token
        if (eosIds.includes(tokenId)) return;
        const text = tokenizer.decodeToken(tokenId, prevToken);
        responseText += text;
        process.stdout.write(`${C.yellow}${text}${C.reset}`);
        prevToken = tokenId;
      },
    });

    const elapsed = Date.now() - genStart;
    log();
    log(`${C.dim}  ${result.generated} tokens | ${result.tokPerSec.toFixed(1)} tok/s | ${elapsed}ms${C.reset}`);
    log();

    // ── Update memory ──
    if (isInstruct) {
      memory.addTurn('user', input);
      memory.addTurn('assistant', responseText.trim());

      // Store notable facts from the conversation
      if (input.length > 15) memory.storeFact(`User said: ${input}`);
      if (responseText.trim().length > 20) memory.storeFact(`Assistant: ${responseText.trim().slice(0, 200)}`);
    }

    // Reset cache for next turn
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
