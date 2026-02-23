/**
 * PureBee Runtime — GPT-2 Transformer
 * 
 * A complete GPT-2 style transformer running entirely on PureBee instructions.
 * No external dependencies. No PyTorch. No CUDA.
 * Just L1 + L2 + L3 executing math.
 * 
 * Architecture:
 *   Token embedding → N x (LayerNorm → Attention → LayerNorm → FFN) → LM Head
 */

'use strict';

const { PureBee } = require('./purebee');

class TransformerConfig {
  constructor(opts = {}) {
    this.vocabSize  = opts.vocabSize  || 50257;   // GPT-2 default
    this.seqLen     = opts.seqLen     || 256;     // max context length
    this.dModel     = opts.dModel     || 64;      // embedding dimension
    this.nHeads     = opts.nHeads     || 4;       // attention heads
    this.nLayers    = opts.nLayers    || 2;       // transformer blocks
    this.dFF        = opts.dFF        || this.dModel * 4;  // feedforward dim
    this.dHead      = this.dModel / this.nHeads;  // per-head dimension
  }
}

class GPTRuntime {
  constructor(config, options = {}) {
    this.config = config;
    this.gpu = new PureBee({ log: options.log || false });
    this._weights = {};
    this._loaded = false;
  }

  /**
   * Initialize model with random weights.
   * In production this would load from a .bin file.
   * Here we prove the architecture runs end to end.
   */
  initRandomWeights() {
    const { vocabSize, dModel, nLayers, dFF } = this.config;
    const gpu = this.gpu;
    const scale = 0.02;

    console.log('  [Runtime] Allocating weights...');

    // Token embedding table  [vocabSize, dModel]
    gpu.GRID_ALLOC('wte', [vocabSize, dModel]).randomize(scale);
    gpu.mem._store.get('wte');  // ensure written

    // Position embedding table  [seqLen, dModel]
    gpu.GRID_ALLOC('wpe', [this.config.seqLen, dModel]).randomize(scale);

    // Final layer norm
    const ln_f_w = gpu.GRID_ALLOC('ln_f.weight', [dModel]);
    ln_f_w.data.fill(1.0);  // initialize to 1
    gpu.GRID_ALLOC('ln_f.bias', [dModel]).fill(0.0);

    // Per-layer weights
    for (let l = 0; l < nLayers; l++) {
      const p = `h${l}`;

      // Layer norm 1
      const ln1w = gpu.GRID_ALLOC(`${p}.ln1.weight`, [dModel]);
      ln1w.data.fill(1.0);
      gpu.GRID_ALLOC(`${p}.ln1.bias`, [dModel]).fill(0.0);

      // Attention projections — Q, K, V combined [dModel, 3*dModel]
      gpu.GRID_ALLOC(`${p}.attn.c_attn.weight`, [dModel, 3 * dModel]).randomize(scale);
      gpu.GRID_ALLOC(`${p}.attn.c_attn.bias`, [3 * dModel]).fill(0.0);

      // Attention output projection [dModel, dModel]
      gpu.GRID_ALLOC(`${p}.attn.c_proj.weight`, [dModel, dModel]).randomize(scale);
      gpu.GRID_ALLOC(`${p}.attn.c_proj.bias`, [dModel]).fill(0.0);

      // Layer norm 2
      const ln2w = gpu.GRID_ALLOC(`${p}.ln2.weight`, [dModel]);
      ln2w.data.fill(1.0);
      gpu.GRID_ALLOC(`${p}.ln2.bias`, [dModel]).fill(0.0);

      // Feed-forward [dModel, dFF] and [dFF, dModel]
      gpu.GRID_ALLOC(`${p}.mlp.c_fc.weight`, [dModel, dFF]).randomize(scale);
      gpu.GRID_ALLOC(`${p}.mlp.c_fc.bias`, [dFF]).fill(0.0);
      gpu.GRID_ALLOC(`${p}.mlp.c_proj.weight`, [dFF, dModel]).randomize(scale);
      gpu.GRID_ALLOC(`${p}.mlp.c_proj.bias`, [dModel]).fill(0.0);
    }

    this._loaded = true;
    const stats = this.gpu.stats();
    console.log(`  [Runtime] ${stats.memory.tensors} tensors, ${stats.memory.totalMB}MB allocated`);
  }

  /**
   * Load weights from a plain JS object (for real model loading).
   * weights = { 'wte': Float32Array, 'wpe': Float32Array, ... }
   */
  loadWeights(weights) {
    const { vocabSize, dModel, nLayers, dFF, seqLen } = this.config;
    const gpu = this.gpu;

    console.log('  [Runtime] Loading weights...');

    for (const [name, data] of Object.entries(weights)) {
      // Determine shape from name and config
      const shape = this._inferShape(name, data.length);
      gpu.GRID_WRITE(name, shape, data instanceof Float32Array ? data : new Float32Array(data));
    }

    this._loaded = true;
    const stats = this.gpu.stats();
    console.log(`  [Runtime] ${stats.memory.tensors} tensors, ${stats.memory.totalMB}MB loaded`);
  }

  _inferShape(name, size) {
    const { vocabSize, dModel, nLayers, dFF, seqLen } = this.config;
    if (name === 'wte') return [vocabSize, dModel];
    if (name === 'wpe') return [seqLen, dModel];
    if (name.endsWith('ln1.weight') || name.endsWith('ln2.weight') || name.endsWith('ln_f.weight')) return [dModel];
    if (name.endsWith('ln1.bias') || name.endsWith('ln2.bias') || name.endsWith('ln_f.bias')) return [dModel];
    if (name.endsWith('c_attn.weight')) return [dModel, 3 * dModel];
    if (name.endsWith('c_attn.bias')) return [3 * dModel];
    if (name.endsWith('c_proj.weight') && name.includes('attn')) return [dModel, dModel];
    if (name.endsWith('c_proj.bias') && name.includes('attn')) return [dModel];
    if (name.endsWith('c_fc.weight')) return [dModel, dFF];
    if (name.endsWith('c_fc.bias')) return [dFF];
    if (name.endsWith('c_proj.weight')) return [dFF, dModel];
    if (name.endsWith('c_proj.bias')) return [dModel];
    // fallback — 1D
    return [size];
  }

  /**
   * Forward pass — run tokens through the transformer.
   * Returns logits [seqLen, vocabSize].
   * 
   * @param {number[]} tokenIds  input token indices
   * @returns {Float32Array}     logits for next token prediction
   */
  forward(tokenIds) {
    if (!this._loaded) throw new Error('Weights not loaded. Call initRandomWeights() or loadWeights().');

    const gpu = this.gpu;
    const { dModel, nLayers, nHeads, dHead, vocabSize } = this.config;
    const seqLen = tokenIds.length;

    // ── EMBEDDING ──
    // Look up token embeddings + position embeddings
    const wte = gpu.GRID_READ('wte');
    const wpe = gpu.GRID_READ('wpe');

    // Build x = wte[tokens] + wpe[positions]  → [seqLen, dModel]
    const xData = new Float32Array(seqLen * dModel);
    for (let i = 0; i < seqLen; i++) {
      const tok = tokenIds[i];
      const tokOffset = tok * dModel;
      const posOffset = i * dModel;
      const xOffset = i * dModel;
      for (let d = 0; d < dModel; d++) {
        xData[xOffset + d] = wte.data[tokOffset + d] + wpe.data[posOffset + d];
      }
    }
    gpu.GRID_WRITE('x', [seqLen, dModel], xData);

    // ── TRANSFORMER BLOCKS ──
    for (let l = 0; l < nLayers; l++) {
      const p = `h${l}`;

      // LayerNorm 1
      gpu.LAYER_NORM('x', `${p}.ln1.weight`, `${p}.ln1.bias`, 'ln1_out');

      // Attention — compute Q, K, V via combined projection
      gpu.LINEAR('ln1_out', `${p}.attn.c_attn.weight`, `${p}.attn.c_attn.bias`, 'qkv');

      // Split QKV → [seqLen, dModel] each
      const qkv = gpu.GRID_READ('qkv');  // [seqLen, 3*dModel]
      const Q_data = new Float32Array(seqLen * dModel);
      const K_data = new Float32Array(seqLen * dModel);
      const V_data = new Float32Array(seqLen * dModel);

      for (let s = 0; s < seqLen; s++) {
        const qkvOffset = s * 3 * dModel;
        const out = s * dModel;
        for (let d = 0; d < dModel; d++) {
          Q_data[out + d] = qkv.data[qkvOffset + d];
          K_data[out + d] = qkv.data[qkvOffset + dModel + d];
          V_data[out + d] = qkv.data[qkvOffset + 2 * dModel + d];
        }
      }

      // Multi-head attention — split across heads, attend, concatenate
      const attnOut = new Float32Array(seqLen * dModel);

      for (let h = 0; h < nHeads; h++) {
        const hOffset = h * dHead;

        // Extract this head's Q, K, V  → [seqLen, dHead]
        const Qh = new Float32Array(seqLen * dHead);
        const Kh = new Float32Array(seqLen * dHead);
        const Vh = new Float32Array(seqLen * dHead);

        for (let s = 0; s < seqLen; s++) {
          const srcOffset = s * dModel + hOffset;
          const dstOffset = s * dHead;
          for (let d = 0; d < dHead; d++) {
            Qh[dstOffset + d] = Q_data[srcOffset + d];
            Kh[dstOffset + d] = K_data[srcOffset + d];
            Vh[dstOffset + d] = V_data[srcOffset + d];
          }
        }

        gpu.GRID_WRITE(`Q_h${h}`, [seqLen, dHead], Qh);
        gpu.GRID_WRITE(`K_h${h}`, [seqLen, dHead], Kh);
        gpu.GRID_WRITE(`V_h${h}`, [seqLen, dHead], Vh);

        // Attention for this head
        gpu.ATTENTION(`Q_h${h}`, `K_h${h}`, `V_h${h}`, `attn_h${h}`, true);
        const headOut = gpu.GRID_READ(`attn_h${h}`);

        // Write back into combined output
        for (let s = 0; s < seqLen; s++) {
          const srcOffset = s * dHead;
          const dstOffset = s * dModel + hOffset;
          for (let d = 0; d < dHead; d++) {
            attnOut[dstOffset + d] = headOut.data[srcOffset + d];
          }
        }
      }

      gpu.GRID_WRITE('attn_concat', [seqLen, dModel], attnOut);

      // Output projection
      gpu.LINEAR('attn_concat', `${p}.attn.c_proj.weight`, `${p}.attn.c_proj.bias`, 'attn_proj');

      // Residual connection: x = x + attn_proj
      gpu.TENSOR_ADD('x', 'attn_proj', 'x_res1');
      gpu.GRID_WRITE('x', [seqLen, dModel], gpu.GRID_READ('x_res1').data);

      // LayerNorm 2
      gpu.LAYER_NORM('x', `${p}.ln2.weight`, `${p}.ln2.bias`, 'ln2_out');

      // Feed-Forward Network
      gpu.LINEAR('ln2_out', `${p}.mlp.c_fc.weight`, `${p}.mlp.c_fc.bias`, 'ff_hidden');
      gpu.GELU('ff_hidden', 'ff_activated');
      gpu.LINEAR('ff_activated', `${p}.mlp.c_proj.weight`, `${p}.mlp.c_proj.bias`, 'ff_out');

      // Residual connection: x = x + ff_out
      gpu.TENSOR_ADD('x', 'ff_out', 'x_res2');
      gpu.GRID_WRITE('x', [seqLen, dModel], gpu.GRID_READ('x_res2').data);
    }

    // ── FINAL LAYER NORM ──
    gpu.LAYER_NORM('x', 'ln_f.weight', 'ln_f.bias', 'x_final');

    // ── LM HEAD — project to vocabulary ──
    // Use weight tying: lm_head = wte^T  [dModel, vocabSize]
    // We compute this as x_final @ wte^T
    const xFinal = gpu.GRID_READ('x_final');
    const wteData = gpu.GRID_READ('wte').data;

    // Last token position only for next-token prediction
    const lastTok = xFinal.data.subarray((seqLen - 1) * dModel, seqLen * dModel);
    const logits = new Float32Array(vocabSize);

    for (let v = 0; v < vocabSize; v++) {
      let dot = 0;
      const wOffset = v * dModel;
      for (let d = 0; d < dModel; d++) {
        dot += lastTok[d] * wteData[wOffset + d];
      }
      logits[v] = dot;
    }

    gpu.SYNC();
    return logits;
  }

  /**
   * Sample next token from logits using top-k sampling.
   * 
   * @param {Float32Array} logits
   * @param {number} topK
   * @param {number} temperature
   * @returns {number} token id
   */
  sample(logits, topK = 40, temperature = 0.8) {
    // Apply temperature
    const scaled = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;

    // Get top-K indices
    const indexed = Array.from(scaled).map((v, i) => [v, i]);
    indexed.sort((a, b) => b[0] - a[0]);
    const topKItems = indexed.slice(0, topK);

    // Softmax over top-K
    const maxV = topKItems[0][0];
    let sum = 0;
    const probs = topKItems.map(([v]) => {
      const e = Math.exp(v - maxV);
      sum += e;
      return e;
    });
    const normalized = probs.map(p => p / sum);

    // Sample
    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < normalized.length; i++) {
      cumulative += normalized[i];
      if (r <= cumulative) return topKItems[i][1];
    }
    return topKItems[0][1];
  }
}

module.exports = { GPTRuntime, TransformerConfig };
