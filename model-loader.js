/**
 * PureBee — 2 — Model Weight Loader
 *
 * Loads real model weights from binary files into PureBee memory.
 * Supports Karpathy's llama2.c binary format (stories15M/42M/110M).
 *
 * Binary format:
 *   Header: 7 x int32 (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
 *   Weights: sequential float32 arrays in fixed order
 *
 * Zero external dependencies. Pure Node.js.
 */

'use strict';

const fs = require('fs');

/**
 * Transpose a row-major matrix [rows, cols] → [cols, rows].
 * Karpathy's format stores weights as [out_dim, in_dim].
 * Our matmul needs x @ W where W is [in_dim, out_dim].
 */
function transpose(data, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[c * rows + r] = data[r * cols + c];
    }
  }
  return out;
}

/**
 * Load a model in Karpathy's llama2.c binary format.
 *
 * @param {string} modelPath - path to .bin file
 * @returns {{ config: Object, weights: Object.<string, Float32Array> }}
 */
function loadKarpathyModel(modelPath) {
  console.log(`  [Loader] Reading ${modelPath}...`);
  const buffer = fs.readFileSync(modelPath);
  const totalBytes = buffer.byteLength;
  console.log(`  [Loader] File size: ${(totalBytes / 1024 / 1024).toFixed(1)}MB`);

  // ── HEADER: 7 int32s = 28 bytes ──
  const headerView = new DataView(buffer.buffer, buffer.byteOffset, 28);
  const config = {
    dim:        headerView.getInt32(0, true),   // little-endian
    hiddenDim:  headerView.getInt32(4, true),
    nLayers:    headerView.getInt32(8, true),
    nHeads:     headerView.getInt32(12, true),
    nKvHeads:   headerView.getInt32(16, true),
    vocabSize:  headerView.getInt32(20, true),
    seqLen:     headerView.getInt32(24, true),
  };

  // Negative vocab_size means unshared weights (separate lm_head)
  const sharedWeights = config.vocabSize > 0;
  config.vocabSize = Math.abs(config.vocabSize);
  config.headDim = Math.floor(config.dim / config.nHeads);
  const kvDim = config.nKvHeads * config.headDim;

  console.log(`  [Loader] Config: dim=${config.dim}, hidden=${config.hiddenDim}, layers=${config.nLayers}, heads=${config.nHeads}, kv_heads=${config.nKvHeads}, vocab=${config.vocabSize}, seq=${config.seqLen}`);

  // ── READ WEIGHTS ──
  // All weights are sequential float32 after the 28-byte header.
  // We use a DataView for safe unaligned reads.
  let byteOffset = 28;
  const weights = {};

  function readFloats(count) {
    const arr = new Float32Array(count);
    const view = new DataView(buffer.buffer, buffer.byteOffset + byteOffset, count * 4);
    for (let i = 0; i < count; i++) {
      arr[i] = view.getFloat32(i * 4, true); // little-endian
    }
    byteOffset += count * 4;
    return arr;
  }

  // Token embedding [vocab_size, dim]
  weights['token_embedding'] = readFloats(config.vocabSize * config.dim);

  // Per-layer RMS attention weights [n_layers * dim] — stored contiguously
  const rmsAttAll = readFloats(config.nLayers * config.dim);

  // wq [n_layers * dim * dim] — stored as [dim, dim] per layer (output, input)
  const wqAll = readFloats(config.nLayers * config.dim * config.dim);

  // wk [n_layers * kvDim * dim] — stored as [kvDim, dim] per layer
  const wkAll = readFloats(config.nLayers * kvDim * config.dim);

  // wv [n_layers * kvDim * dim]
  const wvAll = readFloats(config.nLayers * kvDim * config.dim);

  // wo [n_layers * dim * dim] — stored as [dim, dim] per layer
  const woAll = readFloats(config.nLayers * config.dim * config.dim);

  // Per-layer RMS FFN weights [n_layers * dim]
  const rmsFfnAll = readFloats(config.nLayers * config.dim);

  // w1 [n_layers * hiddenDim * dim] — gate projection, stored as [hiddenDim, dim]
  const w1All = readFloats(config.nLayers * config.hiddenDim * config.dim);

  // w2 [n_layers * dim * hiddenDim] — down projection, stored as [dim, hiddenDim]
  const w2All = readFloats(config.nLayers * config.dim * config.hiddenDim);

  // w3 [n_layers * hiddenDim * dim] — up projection, stored as [hiddenDim, dim]
  const w3All = readFloats(config.nLayers * config.hiddenDim * config.dim);

  // Final RMS norm weight [dim]
  weights['rms_final'] = readFloats(config.dim);

  // If unshared weights, read the classifier/lm_head
  if (!sharedWeights) {
    weights['lm_head'] = readFloats(config.vocabSize * config.dim);
  }

  console.log(`  [Loader] Read ${(byteOffset / 1024 / 1024).toFixed(1)}MB of ${(totalBytes / 1024 / 1024).toFixed(1)}MB`);

  // ── SPLIT PER-LAYER AND TRANSPOSE ──
  // Our matmul: y = x @ W, where x is [seq, in_dim], W is [in_dim, out_dim]
  // File stores W as [out_dim, in_dim], so we transpose to [in_dim, out_dim]

  for (let l = 0; l < config.nLayers; l++) {
    const ld = config.dim;
    const lh = config.hiddenDim;

    // RMS attention weight — [dim], no transpose needed
    weights[`layer${l}.rms_att`] = rmsAttAll.slice(l * ld, (l + 1) * ld);

    // wq: file [dim, dim] → transpose to [dim, dim] for x @ wq
    const wqSlice = wqAll.slice(l * ld * ld, (l + 1) * ld * ld);
    weights[`layer${l}.wq`] = transpose(wqSlice, ld, ld);

    // wk: file [kvDim, dim] → transpose to [dim, kvDim]
    const wkSlice = wkAll.slice(l * kvDim * ld, (l + 1) * kvDim * ld);
    weights[`layer${l}.wk`] = transpose(wkSlice, kvDim, ld);

    // wv: file [kvDim, dim] → transpose to [dim, kvDim]
    const wvSlice = wvAll.slice(l * kvDim * ld, (l + 1) * kvDim * ld);
    weights[`layer${l}.wv`] = transpose(wvSlice, kvDim, ld);

    // wo: file [dim, dim] → transpose to [dim, dim]
    const woSlice = woAll.slice(l * ld * ld, (l + 1) * ld * ld);
    weights[`layer${l}.wo`] = transpose(woSlice, ld, ld);

    // RMS FFN weight — [dim], no transpose
    weights[`layer${l}.rms_ffn`] = rmsFfnAll.slice(l * ld, (l + 1) * ld);

    // w1 (gate): file [hiddenDim, dim] → transpose to [dim, hiddenDim]
    const w1Slice = w1All.slice(l * lh * ld, (l + 1) * lh * ld);
    weights[`layer${l}.w1`] = transpose(w1Slice, lh, ld);

    // w2 (down): file [dim, hiddenDim] → transpose to [hiddenDim, dim]
    const w2Slice = w2All.slice(l * ld * lh, (l + 1) * ld * lh);
    weights[`layer${l}.w2`] = transpose(w2Slice, ld, lh);

    // w3 (up): file [hiddenDim, dim] → transpose to [dim, hiddenDim]
    const w3Slice = w3All.slice(l * lh * ld, (l + 1) * lh * ld);
    weights[`layer${l}.w3`] = transpose(w3Slice, lh, ld);
  }

  // Count total parameters
  let totalParams = 0;
  for (const [name, data] of Object.entries(weights)) {
    if (!name.startsWith('_')) totalParams += data.length;
  }

  console.log(`  [Loader] ${Object.keys(weights).length} weight tensors, ${(totalParams / 1e6).toFixed(1)}M parameters`);
  console.log(`  [Loader] Shared weights: ${sharedWeights}`);

  return { config, weights, sharedWeights };
}

module.exports = { loadKarpathyModel, transpose };
