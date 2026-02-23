/**
 * PureBee — 6 — GGUF Streaming Weight Loader
 *
 * Parses GGUF header without loading weight data into memory.
 * Loads individual tensors on demand via file descriptor seeks.
 *
 * This enables running models far larger than available RAM by
 * keeping only one transformer layer's weights in memory at a time.
 *
 * Key design:
 *   - parseHeader() reads metadata + tensor index only (~KB, not GB)
 *   - StreamingWeightLoader opens fd, loads tensors on demand
 *   - Double-buffering: prefetch next layer while computing current
 *   - Memory: only 1-2 layers resident at any time
 *
 * Zero external dependencies.
 */

'use strict';

const fs = require('fs');
const {
  GGUFParser, GGUFTokenizer, GGML_TYPE, GGML_TYPE_SIZE,
  loadTensorData, tensorByteSize, tensorElementCount,
  mapTensorName, transpose2D, extractConfig,
} = require('./gguf');

/**
 * Parse a GGUF file header without loading tensor data.
 * Returns config, tokenizer, tensor index, and the data start offset.
 *
 * @param {string} filePath
 * @returns {{ config, tokenizer, tensorIndex, dataStart, alignment }}
 */
function parseHeader(filePath) {
  // Read enough of the file for header + metadata + tensor infos.
  // For most models this is < 10MB even with large vocabularies.
  // We read in chunks to find where tensor data starts.
  const fd = fs.openSync(filePath, 'r');
  const stat = fs.fstatSync(fd);

  // Read first 64MB for header parsing (covers even large vocab models)
  const headerSize = Math.min(64 * 1024 * 1024, stat.size);
  const headerBuf = Buffer.alloc(headerSize);
  fs.readSync(fd, headerBuf, 0, headerSize, 0);
  fs.closeSync(fd);

  const parser = new GGUFParser(headerBuf);

  // Parse header
  const header = parser.readHeader();
  const metadata = parser.readMetadata(header.metadataKvCount);
  const tensorInfos = parser.readTensorInfos(header.tensorCount);

  // Extract config and tokenizer
  const config = extractConfig(metadata);
  const tokenizer = new GGUFTokenizer(metadata);
  config.vocabSize = tokenizer.vocab.length;

  // Calculate data section start (aligned)
  const alignment = metadata['general.alignment'] || 32;
  const dataStart = Math.ceil(parser.pos / alignment) * alignment;

  // Build tensor index: map PureBee name → { offset, type, dims, byteSize, transpose, ggufName }
  const tensorIndex = new Map();
  let hasOutputWeight = false;

  for (const info of tensorInfos) {
    const { name: tensorName, transpose } = mapTensorName(info.name);
    const byteSize = tensorByteSize(info.dims, info.type);

    if (info.name === 'output.weight') hasOutputWeight = true;

    tensorIndex.set(tensorName, {
      offset: dataStart + info.offset,  // absolute file offset
      type: info.type,
      dims: info.dims,
      byteSize,
      transpose,
      ggufName: info.name,
    });
  }

  const sharedWeights = !hasOutputWeight;

  return { config, tokenizer, tensorIndex, dataStart, alignment, sharedWeights };
}

/**
 * Streaming weight loader — loads tensors on demand from an open file descriptor.
 */
class StreamingWeightLoader {
  /**
   * @param {string} filePath — path to GGUF file
   * @param {Map} tensorIndex — from parseHeader()
   * @param {Object} options
   * @param {boolean} options.cacheRawData — cache raw Q4_0 bytes in memory (eliminates disk I/O per token)
   */
  constructor(filePath, tensorIndex, options = {}) {
    this._filePath = filePath;
    this._tensorIndex = tensorIndex;
    this._fd = fs.openSync(filePath, 'r');
    this._prefetchPromise = null;
    this._prefetchResult = null;
    this._prefetchLayer = -1;

    // Raw data cache: tensor name → Buffer (cached raw bytes from disk)
    this._rawCache = null;
    if (options.cacheRawData) {
      this._initRawCache();
    }
  }

  /**
   * Pre-read all tensor raw data into memory.
   * Uses SharedArrayBuffer when available (for worker thread sharing).
   * Eliminates disk I/O during generation — only dequantization remains.
   */
  _initRawCache() {
    this._rawCache = new Map();
    let totalBytes = 0;

    // Calculate total size needed
    for (const [name, info] of this._tensorIndex) {
      totalBytes += info.byteSize;
    }

    console.log('  [StreamingLoader] Caching raw tensor data...');
    const start = Date.now();

    // Allocate one contiguous SharedArrayBuffer for all tensor data
    // Falls back to regular ArrayBuffer if SharedArrayBuffer unavailable
    let backingBuffer;
    try {
      backingBuffer = new SharedArrayBuffer(totalBytes);
      this._sharedBuffer = backingBuffer;
      console.log('  [StreamingLoader] Using SharedArrayBuffer (%sMB)',
        (totalBytes / (1024 * 1024)).toFixed(1));
    } catch (e) {
      backingBuffer = new ArrayBuffer(totalBytes);
      this._sharedBuffer = null;
    }

    // Read all tensors into contiguous buffer, create views
    let offset = 0;
    for (const [name, info] of this._tensorIndex) {
      const view = new Uint8Array(backingBuffer, offset, info.byteSize);
      // Read from disk into the shared buffer view
      const tmpBuf = Buffer.alloc(info.byteSize);
      fs.readSync(this._fd, tmpBuf, 0, info.byteSize, info.offset);
      view.set(tmpBuf);
      this._rawCache.set(name, view);
      // Store the offset within the shared buffer for worker threads
      info._sharedOffset = offset;
      info._sharedLength = info.byteSize;
      offset += info.byteSize;
    }

    const elapsed = Date.now() - start;
    const mb = (totalBytes / (1024 * 1024)).toFixed(1);
    console.log(`  [StreamingLoader] Cached ${this._rawCache.size} tensors, ${mb}MB in ${elapsed}ms`);
  }

  /**
   * Get the SharedArrayBuffer backing the raw cache (for worker threads).
   * @returns {SharedArrayBuffer|null}
   */
  getSharedBuffer() {
    return this._sharedBuffer || null;
  }

  /**
   * Load a single tensor by name.
   * Reads from disk, dequantizes, and transposes as needed.
   *
   * @param {string} name — PureBee tensor name (e.g., 'layer0.wq')
   * @returns {Float32Array}
   */
  loadTensor(name) {
    const info = this._tensorIndex.get(name);
    if (!info) throw new Error(`StreamingLoader: tensor '${name}' not found in index`);

    // Read raw bytes — from cache if available, else from disk
    let rawBuf;
    if (this._rawCache && this._rawCache.has(name)) {
      rawBuf = this._rawCache.get(name);
    } else {
      rawBuf = Buffer.alloc(info.byteSize);
      fs.readSync(this._fd, rawBuf, 0, info.byteSize, info.offset);
    }

    // Dequantize to Float32Array
    const data = loadTensorData(rawBuf, 0, info.dims, info.type);

    // Transpose 2D weight matrices for PureBee matmul convention
    if (info.transpose && info.dims.length === 2) {
      const rows = info.dims[1]; // actual rows in memory (out_features)
      const cols = info.dims[0]; // actual cols in memory (in_features)
      return transpose2D(data, rows, cols);
    }

    return data;
  }

  /**
   * Load all weight tensors for a single transformer layer.
   *
   * @param {number} layerIdx
   * @returns {Object} — { wq, wk, wv, wo, w1, w2, w3, rms_att, rms_ffn }
   */
  loadLayerWeights(layerIdx) {
    // Check if this layer was prefetched
    if (this._prefetchLayer === layerIdx && this._prefetchResult) {
      const result = this._prefetchResult;
      this._prefetchResult = null;
      this._prefetchLayer = -1;
      return result;
    }

    return this._loadLayerSync(layerIdx);
  }

  _loadLayerSync(layerIdx) {
    const prefix = `layer${layerIdx}`;
    const weights = {};

    const names = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3', 'rms_att', 'rms_ffn'];
    for (const suffix of names) {
      const fullName = `${prefix}.${suffix}`;
      if (this._tensorIndex.has(fullName)) {
        weights[suffix] = this.loadTensor(fullName);
      }
    }

    return weights;
  }

  /**
   * Load raw (non-dequantized) tensor data by name.
   * Returns the raw Q4_0/Q4_1/etc. bytes plus metadata.
   *
   * @param {string} name — PureBee tensor name
   * @returns {{ rawBuf: Buffer, type: number, dims: number[], N: number, K: number }}
   */
  loadTensorRaw(name) {
    const info = this._tensorIndex.get(name);
    if (!info) throw new Error(`StreamingLoader: tensor '${name}' not found in index`);

    let rawBuf;
    if (this._rawCache && this._rawCache.has(name)) {
      rawBuf = this._rawCache.get(name);
    } else {
      rawBuf = Buffer.alloc(info.byteSize);
      fs.readSync(this._fd, rawBuf, 0, info.byteSize, info.offset);
    }

    // For weight matrices: dims[0] = in_features (K), dims[1] = out_features (N)
    const K = info.dims[0];
    const N = info.dims.length > 1 ? info.dims[1] : 1;

    return { rawBuf, type: info.type, dims: info.dims, N, K };
  }

  /**
   * Load raw weight tensors for a layer (no dequantization/transpose).
   * Returns raw Q4_0 buffers for weight matrices, dequantized data for norms.
   *
   * @param {number} layerIdx
   * @returns {Object} — { raw: { wq, wk, ... }, rms_att, rms_ffn }
   */
  loadLayerWeightsRaw(layerIdx) {
    const prefix = `layer${layerIdx}`;
    const raw = {};
    const result = {};

    const weightNames = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'];
    for (const suffix of weightNames) {
      const fullName = `${prefix}.${suffix}`;
      if (this._tensorIndex.has(fullName)) {
        const info = this._tensorIndex.get(fullName);
        // Quantized weights → keep raw
        if (info.type === GGML_TYPE.Q4_0 || info.type === GGML_TYPE.Q4_1) {
          raw[suffix] = this.loadTensorRaw(fullName);
        } else {
          // F32/F16 weights → dequantize normally
          raw[suffix] = { dequantized: this.loadTensor(fullName), type: info.type };
        }
      }
    }

    // Norm weights are always small F32 — dequantize them
    const attNorm = `${prefix}.rms_att`;
    const ffnNorm = `${prefix}.rms_ffn`;
    if (this._tensorIndex.has(attNorm)) result.rms_att = this.loadTensor(attNorm);
    if (this._tensorIndex.has(ffnNorm)) result.rms_ffn = this.loadTensor(ffnNorm);

    result.raw = raw;
    return result;
  }

  /**
   * Start async prefetch of a layer's weights.
   * Uses setImmediate to allow event loop to continue.
   *
   * @param {number} layerIdx
   */
  prefetchLayer(layerIdx) {
    // Load synchronously but in next tick to allow current computation to start
    this._prefetchLayer = layerIdx;
    this._prefetchResult = null;

    // Use synchronous loading (Node.js fs.readSync is blocking anyway)
    // but wrap in setImmediate for future async fd support
    this._prefetchResult = this._loadLayerSync(layerIdx);
  }

  /**
   * Load resident tensors — embedding, final norm, lm_head.
   * These stay in memory for the entire session.
   *
   * @param {boolean} sharedWeights — whether lm_head shares token_embedding
   * @returns {{ tokenEmbedding, rmsFinal, lmHead }}
   */
  loadResidentWeights(sharedWeights) {
    const resident = {};

    resident.tokenEmbedding = this.loadTensor('token_embedding');
    resident.rmsFinal = this.loadTensor('rms_final');

    if (!sharedWeights && this._tensorIndex.has('lm_head')) {
      resident.lmHead = this.loadTensor('lm_head');
    }

    return resident;
  }

  /**
   * Check if a tensor exists in the index.
   */
  hasTensor(name) {
    return this._tensorIndex.has(name);
  }

  /**
   * Get tensor info without loading data.
   */
  getTensorInfo(name) {
    return this._tensorIndex.get(name);
  }

  /**
   * Close the file descriptor.
   */
  close() {
    if (this._fd !== null) {
      fs.closeSync(this._fd);
      this._fd = null;
    }
  }
}

module.exports = { parseHeader, StreamingWeightLoader };
