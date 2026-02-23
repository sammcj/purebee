/**
 * PureBee L3 — Instruction Set
 *
 * The defined contract between the outside world and PureBee.
 * Every operation goes through here. The memory model and execution
 * engine are internal — this is the public API.
 *
 * Instructions:
 *   GRID_ALLOC    — allocate a tensor
 *   GRID_WRITE    — write data into a tensor
 *   GRID_READ     — read a tensor
 *   TENSOR_MUL    — matrix multiplication
 *   TENSOR_ADD    — element-wise addition
 *   SOFTMAX       — probability normalization
 *   LAYER_NORM    — layer normalization
 *   GELU          — activation function
 *   ATTENTION     — scaled dot-product attention
 *   SYNC          — barrier: ensure all ops complete
 */

'use strict';

const { PureBeeMemory, Tensor } = require('./memory');
const { ExecutionEngine } = require('./engine');
const { QuantizedTensor } = require('./quantize');

class PureBee {
  constructor(options = {}) {
    this.mem = new PureBeeMemory();
    this.engine = new ExecutionEngine();
    this._log = options.log || false;
    this._opLog = [];
  }

  _emit(instruction, details) {
    if (this._log) {
      console.log(`  [PureBee] ${instruction}${details ? ' ' + details : ''}`);
    }
    this._opLog.push({ instruction, details, t: Date.now() });
  }

  /** GRID_ALLOC — allocate a new empty tensor */
  GRID_ALLOC(name, shape) {
    this._emit('GRID_ALLOC', `${name} [${shape}]`);
    return this.mem.alloc(name, shape);
  }

  /** GRID_WRITE — write Float32Array or plain array into a tensor */
  GRID_WRITE(name, shape, data) {
    this._emit('GRID_WRITE', `${name} [${shape}]`);
    return this.mem.write(name, shape, data);
  }

  /** GRID_READ — read a tensor by name */
  GRID_READ(name) {
    this._emit('GRID_READ', name);
    return this.mem.read(name);
  }

  /** TENSOR_MUL — matrix multiply A [M,K] x B [K,N] → C [M,N] */
  TENSOR_MUL(aName, bName, outName) {
    this._emit('TENSOR_MUL', `${aName} x ${bName} → ${outName}`);
    const A = this.mem.read(aName);
    const B = this.mem.read(bName);
    const result = this.engine.tensorMul(A, B, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /**
   * LINEAR — fused matmul + bias (y = x @ W + b)
   * Auto-detects quantized weights and dispatches to Q8 matmul if applicable.
   */
  LINEAR(xName, wName, bName, outName) {
    this._emit('LINEAR', `${xName} @ ${wName} + ${bName} → ${outName}`);
    const x = this.mem.read(xName);
    const W = this._readRaw(wName); // may be Tensor or QuantizedTensor
    const b = bName ? this.mem.read(bName) : null;

    let result;
    if (W instanceof QuantizedTensor) {
      result = this.engine.tensorMulAddQ8(x, W, b, outName);
    } else {
      result = this.engine.tensorMulAdd(x, W, b, outName);
    }
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** Store a QuantizedTensor or any raw object in PureBee memory */
  GRID_WRITE_RAW(name, obj) {
    this._emit('GRID_WRITE_RAW', name);
    this.mem._store.set(name, obj);
    return obj;
  }

  /** Read from memory — returns whatever is stored (Tensor, QuantizedTensor, etc.) */
  _readRaw(name) {
    const obj = this.mem._store.get(name);
    if (!obj) throw new Error(`PureBee: '${name}' not found in memory`);
    return obj;
  }

  /** TENSOR_ADD — element-wise add */
  TENSOR_ADD(aName, bName, outName) {
    this._emit('TENSOR_ADD', `${aName} + ${bName} → ${outName}`);
    const A = this.mem.read(aName);
    const B = this.mem.read(bName);
    const result = this.engine.tensorAdd(A, B, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** SOFTMAX — normalize logits to probabilities */
  SOFTMAX(inputName, outName) {
    this._emit('SOFTMAX', `${inputName} → ${outName}`);
    const input = this.mem.read(inputName);
    const result = this.engine.softmax(input, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** LAYER_NORM — normalize + scale + shift */
  LAYER_NORM(xName, weightName, biasName, outName, eps = 1e-5) {
    this._emit('LAYER_NORM', `${xName} → ${outName}`);
    const x = this.mem.read(xName);
    const weight = this.mem.read(weightName);
    const bias = biasName ? this.mem.read(biasName) : null;
    const result = this.engine.layerNorm(x, weight, bias, eps, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** GELU — activation function */
  GELU(inputName, outName) {
    this._emit('GELU', `${inputName} → ${outName}`);
    const input = this.mem.read(inputName);
    const result = this.engine.gelu(input, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** ATTENTION — scaled dot-product attention */
  ATTENTION(qName, kName, vName, outName, causal = true) {
    this._emit('ATTENTION', `Q=${qName} K=${kName} V=${vName} → ${outName}`);
    const Q = this.mem.read(qName);
    const K = this.mem.read(kName);
    const V = this.mem.read(vName);
    const result = this.engine.attention(Q, K, V, causal, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** PARALLEL_MAP — apply a function element-wise to a tensor */
  PARALLEL_MAP(inputName, fn, outName) {
    this._emit('PARALLEL_MAP', `${inputName} → ${outName}`);
    const input = this.mem.read(inputName);
    const result = this.engine.parallelMap(input, fn, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** RMS_NORM — root mean square normalization (LLaMA) */
  RMS_NORM(xName, weightName, outName, eps = 1e-5) {
    this._emit('RMS_NORM', `${xName} → ${outName}`);
    const x = this.mem.read(xName);
    const weight = this.mem.read(weightName);
    const result = this.engine.rmsNorm(x, weight, eps, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** SILU — sigmoid linear unit activation (LLaMA) */
  SILU(inputName, outName) {
    this._emit('SILU', `${inputName} → ${outName}`);
    const input = this.mem.read(inputName);
    const result = this.engine.silu(input, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** ELEMENT_MUL — element-wise multiplication */
  ELEMENT_MUL(aName, bName, outName) {
    this._emit('ELEMENT_MUL', `${aName} * ${bName} → ${outName}`);
    const A = this.mem.read(aName);
    const B = this.mem.read(bName);
    const result = this.engine.elementMul(A, B, outName);
    this.mem.write(outName, result.shape, result.data);
    return result;
  }

  /** SYNC — barrier, no-op in single-threaded mode but required by spec */
  SYNC() {
    this._emit('SYNC');
    // In a multi-threaded future, this would flush all pending ops
    return true;
  }

  /** Free a tensor from memory */
  FREE(name) {
    this.mem.free(name);
  }

  stats() {
    return {
      memory: this.mem.stats(),
      engine: this.engine.stats,
      ops: this._opLog.length
    };
  }
}

module.exports = { PureBee };
