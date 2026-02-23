/**
 * PureBee L1 — Memory Model
 *
 * The grid. Every tensor in PureBee lives here.
 * A tensor is a named, typed, shaped block of Float32 values.
 * This is VRAM — except it lives in process memory.
 */

'use strict';

class Tensor {
  /**
   * @param {string} name
   * @param {number[]} shape  e.g. [512, 512] or [1, 32, 64]
   * @param {Float32Array} [data]
   */
  constructor(name, shape, data = null) {
    this.name = name;
    this.shape = shape;
    this.size = shape.reduce((a, b) => a * b, 1);
    this.data = data || new Float32Array(this.size);
  }

  /** Flat index from multi-dimensional coordinates */
  index(...coords) {
    let idx = 0;
    let stride = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      idx += coords[i] * stride;
      stride *= this.shape[i];
    }
    return idx;
  }

  /** Read a single value */
  get(...coords) {
    return this.data[this.index(...coords)];
  }

  /** Write a single value */
  set(value, ...coords) {
    this.data[this.index(...coords)] = value;
  }

  /** Fill all cells with a value */
  fill(value) {
    this.data.fill(value);
    return this;
  }

  /** Fill with random values in range [-scale, scale] */
  randomize(scale = 0.02) {
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] = (Math.random() * 2 - 1) * scale;
    }
    return this;
  }

  /** Return a view of a slice along dim=0 (row slice) */
  row(i) {
    const rowSize = this.size / this.shape[0];
    const start = i * rowSize;
    return this.data.subarray(start, start + rowSize);
  }

  /** Clone this tensor */
  clone(name) {
    const t = new Tensor(name || this.name + '_clone', [...this.shape]);
    t.data.set(this.data);
    return t;
  }

  get bytes() {
    return this.data.byteLength;
  }

  toString() {
    const mb = (this.bytes / 1024 / 1024).toFixed(2);
    return `Tensor(${this.name}, shape=[${this.shape}], size=${this.size}, ${mb}MB)`;
  }
}

/**
 * PureBee Memory — the grid registry.
 * All tensors allocated by PureBee live here.
 */
class PureBeeMemory {
  constructor() {
    this._store = new Map();
    this._totalBytes = 0;
  }

  /** Allocate a new empty tensor */
  alloc(name, shape) {
    const t = new Tensor(name, shape);
    this._store.set(name, t);
    this._totalBytes += t.bytes;
    return t;
  }

  /** Write data into an existing or new tensor. Reallocates if shape changes. */
  write(name, shape, data) {
    const newSize = shape.reduce((a, b) => a * b, 1);
    let t = this._store.get(name);
    if (!t || t.size !== newSize) {
      // Free old if exists
      if (t) this._totalBytes -= t.bytes;
      t = new Tensor(name, shape);
      this._store.set(name, t);
      this._totalBytes += t.bytes;
    } else {
      // Update shape metadata in case dims changed but size same
      t.shape = shape;
    }
    if (data instanceof Float32Array) {
      t.data.set(data);
    } else if (Array.isArray(data)) {
      for (let i = 0; i < data.length; i++) t.data[i] = data[i];
    }
    return t;
  }

  /** Read a tensor by name */
  read(name) {
    const t = this._store.get(name);
    if (!t) throw new Error(`PureBee: tensor '${name}' not found in memory`);
    return t;
  }

  /** Check if a tensor exists */
  has(name) {
    return this._store.has(name);
  }

  /** Free a tensor */
  free(name) {
    const t = this._store.get(name);
    if (t) {
      this._totalBytes -= t.bytes;
      this._store.delete(name);
    }
  }

  get totalMB() {
    return (this._totalBytes / 1024 / 1024).toFixed(1);
  }

  stats() {
    return {
      tensors: this._store.size,
      totalMB: this.totalMB,
      keys: [...this._store.keys()]
    };
  }
}

module.exports = { Tensor, PureBeeMemory };
