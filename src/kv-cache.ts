import {core as mx, nn} from '@frost-beta/mlx';
import type {BaseModel} from './base.js';

/**
 * The base class of KV cache.
 */
export abstract class BaseKVCache {
  keys?: mx.array;
  values?: mx.array;
  offset = 0;
  step = 256;

  static createForModel<T extends BaseKVCache>(
      model: BaseModel,
      construct: new (headDim: number, nKVHeads: number) => T) {
    const cache: BaseKVCache[] = [];
    for (let i = 0; i < model.layers.length; ++i)
      cache[i] = new construct(model.headDim, model.nKVHeads);
    return cache;
  }

  abstract updateAndFetch(keys: mx.array, values: mx.array): [ mx.array, mx.array ];

  get state() {
    return [ this.keys, this.values ];
  }
}

/**
 * A design of KV cache friendly to MLX's memory cache design, which allocates
 * arrays in same shapes.
 *
 * See also https://github.com/ml-explore/mlx-examples/issues/724.
 */
export class KVCache extends BaseKVCache {
  constructor(public headDim: number,
              public nKVHeads: number) {
    super();
  }

  static override createForModel(model: BaseModel) {
    return BaseKVCache.createForModel<KVCache>(model, KVCache);
  }

  override updateAndFetch(keys: mx.array, values: mx.array): [ mx.array, mx.array ] {
    const prev = this.offset;
    if (!this.keys || (prev + keys.shape[2] > this.keys.shape[2])) {
      const B = keys.shape[0];
      const nSteps = Math.floor((this.step + keys.shape[2] - 1) / this.step);
      const shape = [ B, this.nKVHeads, nSteps * this.step, this.headDim ];
      const newK = mx.zeros(shape, keys.dtype);
      const newV = mx.zeros(shape, values.dtype);
      if (this.keys) {
        const old = [ this.keys, this.values ];
        if (prev % this.step != 0) {
          const get: mx.ArrayIndex[] = [ '...', mx.Slice(null, prev), mx.Slice() ];
          this.keys = this.keys.index(...get);
          this.values = this.values.index(...get);
        }
        this.keys = mx.concatenate([ this.keys, newK ], 2);
        this.values = mx.concatenate([ this.values, newV ], 2);
        mx.dispose(old);
      } else {
        this.keys = newK;
        this.values = newV;
      }
    }

    this.offset += keys.shape[2];

    const insert: mx.ArrayIndex[] = [ '...', mx.Slice(prev, this.offset), mx.Slice() ];
    this.keys.indexPut_(insert, keys);
    this.values.indexPut_(insert, values);

    const get: mx.ArrayIndex[] = [ '...', mx.Slice(null, this.offset), mx.Slice() ];
    return [ this.keys.index(...get), this.values.index(...get) ];
  }
}

/**
 * KV cache using rotating buffer, enabling infinite generations.
 *
 * See also https://github.com/ml-explore/mlx-examples/pull/931.
 */
export class RotatingKVCache extends BaseKVCache {
  kHeadDim: number;
  vHeadDim: number;
  #idx = 0;

  static override createForModel(model: BaseModel) {
    return BaseKVCache.createForModel(model, RotatingKVCache);
  }

  constructor(headDim: number,
              public nKVHeads: number,
              public maxSize = 1024,
              public keep = 4) {
    super();
    this.kHeadDim = this.vHeadDim = headDim;
  }

  override updateAndFetch(keys: mx.array, values: mx.array): [ mx.array, mx.array ] {
    const prev = this.offset;
    const [ B, , S ] = keys.shape;

    // Prefill mode.
    if (S > 1) {
      if (!this.keys) {
        this.keys = keys;
        this.values = values;
      } else {
        // The largest size is this.maxSize + S - 1 to ensure every token gets
        // at least this.maxSize context.
        const trimSize = this.keys.shape[2] - this.maxSize + 1;
        const old = [ this.keys, this.values ];
        this.keys = this.trim(trimSize, this.keys, keys);
        this.values = this.trim(trimSize, this.values, values);
        mx.dispose(old);
      }
      this.offset += S;
      this.#idx = this.keys.shape[2];
      return [ this.keys, this.values ];
    }

    // Generation mode.

    // May not have hit the max size yet, so potentiall keep growing the cache.
    if (!this.keys || (prev >= this.keys.shape[2] && this.keys.shape[2] < this.maxSize)) {
      const newSize = Math.min(this.step, this.maxSize - prev);
      const kShape = [ B, this.nKVHeads, newSize, this.kHeadDim ];
      const vShape = [ B, this.nKVHeads, newSize, this.vHeadDim ];
      const newK = mx.zeros(kShape, keys.dtype);
      const newV = mx.zeros(vShape, values.dtype);
      if (this.keys) {
        const old = [ this.keys, this.values ];
        this.keys = mx.concatenate([ this.keys, newK ], 2);
        this.values = mx.concatenate([ this.values, newV ], 2);
        mx.dispose(old);
      } else {
        this.keys = newK;
        this.values = newV;
      }
      this.#idx = prev;
    }

    // Trim if needed.
    const trimSize = this.keys.shape[2] - this.maxSize;
    if (trimSize > 0) {
      const old = [ this.keys, this.values ];
      this.keys = this.trim(trimSize, this.keys);
      this.values = this.trim(trimSize, this.values);
      mx.dispose(old);
      this.#idx = this.maxSize;
    }

    // Rotate.
    if (this.#idx == this.maxSize) {
      this.#idx = this.keep;
    }

    // Assign.
    const insert: mx.ArrayIndex[] = [ '...', mx.Slice(this.#idx, this.#idx + 1), mx.Slice() ];
    this.keys.indexPut_(insert, keys);
    this.values.indexPut_(insert, values);
    this.offset += 1;
    this.#idx += 1;

    // If the buffer is not full, slice off the end.
    if (this.offset < this.maxSize) {
      const get: mx.ArrayIndex[] = [ '...', mx.Slice(null, this.offset), mx.Slice() ];
      return [ this.keys.index(...get), this.values.index(...get) ];
    }
    return [ this.keys, this.values ];
  }

  private trim(trimSize: number, v: mx.array, append?: mx.array) {
    let toCat: mx.array[];
    if (trimSize > 0) {
      toCat = [ v.index('...', mx.Slice(0, this.keep), mx.Slice()),
                v.index('...', mx.Slice(trimSize + this.keep), mx.Slice()) ];
    } else {
      toCat = [ v ];
    }
    if (append) {
      toCat.push(append);
    }
    return mx.concatenate(toCat, 2);
  }
}
