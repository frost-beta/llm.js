import {core as mx, nn} from '@frost-beta/mlx';

export interface KVCacheOptions {
  nLayers: number;
}

/**
 * The base class of KV cache.
 */
export abstract class BaseKVCache {
  keys?: mx.array;
  values?: mx.array;
  step = 256;

  static create<T extends BaseKVCache>(options: KVCacheOptions,
                                       construct: new (...args: any[]) => T,
                                       ...args: any[]) {
    const cache: BaseKVCache[] = [];
    for (let i = 0; i < options.nLayers; ++i)
      cache[i] = new construct(...args);
    return cache;
  }

  abstract updateAndFetch(keys: mx.array, values: mx.array): [ mx.array, mx.array ];
  abstract get state(): any;
  abstract get offset(): number;
}

/**
 * A design of KV cache friendly to MLX's memory cache design, which allocates
 * arrays in same shapes.
 *
 * See also https://github.com/ml-explore/mlx-examples/issues/724.
 */
export class KVCache extends BaseKVCache {
  #offset = 0;

  static override create(options: KVCacheOptions) {
    return BaseKVCache.create<KVCache>(options, KVCache);
  }

  override updateAndFetch(keys: mx.array, values: mx.array): [ mx.array, mx.array ] {
    const prev = this.#offset;
    if (!this.keys || (prev + keys.shape[2] > this.keys.shape[2])) {
      const [ B, nKVHeads, , kHeadDim ] = keys.shape;
      const vHeadDim = values.shape[3];
      const nSteps = Math.floor((this.step + keys.shape[2] - 1) / this.step);
      const kShape = [ B, nKVHeads, nSteps * this.step, kHeadDim ];
      const vShape = [ B, nKVHeads, nSteps * this.step, vHeadDim ];
      const newK = mx.zeros(kShape, keys.dtype);
      const newV = mx.zeros(vShape, values.dtype);
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

    this.#offset += keys.shape[2];

    const insert: mx.ArrayIndex[] = [ '...', mx.Slice(prev, this.#offset), mx.Slice() ];
    this.keys.indexPut_(insert, keys);
    this.values.indexPut_(insert, values);

    const get: mx.ArrayIndex[] = [ '...', mx.Slice(null, this.#offset), mx.Slice() ];
    return [ this.keys.index(...get), this.values.index(...get) ];
  }

  override get state() {
    if (this.#offset == this.keys.shape[2]) {
      return [ this.keys, this.values ];
    } else {
      const get: mx.ArrayIndex[] = [ '...', mx.Slice(null, this.#offset), mx.Slice() ];
      return [ this.keys.index(...get), this.values.index(...get) ];
    }
  }

  override get offset() {
    return this.#offset;
  }
}

/**
 * KV cache using rotating buffer, enabling infinite generations.
 *
 * See also https://github.com/ml-explore/mlx-examples/pull/931.
 */
export class RotatingKVCache extends BaseKVCache {
  #offset = 0;
  #idx = 0;

  static override create(options: KVCacheOptions, ...args: any[]) {
    return BaseKVCache.create(options, RotatingKVCache, ...args);
  }

  constructor(public maxSize = 1024, public keep = 4) {
    super();
  }

  override updateAndFetch(keys: mx.array, values: mx.array): [ mx.array, mx.array ] {
    if (keys.shape[2] == 1)
      return this.updateInPlace(keys, values);
    else
      return this.updateConcat(keys, values);
  }

  override get state() {
    if (this.#offset < this.keys.shape[2]) {
      const get: mx.ArrayIndex[] = [ '...', mx.Slice(null, this.#offset), mx.Slice() ];
      return [ this.keys.index(...get), this.values.index(...get) ];
    } else {
      return [ this.keys, this.values ];
    }
  }

  override get offset(): number {
    return Math.min(this.maxSize - 1, this.#offset);
  }

  private updateConcat(keys: mx.array, values: mx.array): [ mx.array, mx.array ] {
    if (!this.keys) {
      this.keys = keys;
      this.values = values;
    } else {
      // Put the keys/values in temporal order to preserve context.
      const old = [ this.keys, this.values ];
      this.keys = this.temporalOrder(this.keys);
      this.values = this.temporalOrder(this.values);

      // The largest size is self.max_size + S to ensure every token gets
      // at least this.maxSize context.
      const trimSize = this.#idx - this.maxSize;
      this.keys = this.trim(trimSize, this.keys, keys);
      this.values = this.trim(trimSize, this.values, values);
      mx.dispose(old);
    }
    this.#offset += keys.shape[2];
    this.#idx = this.keys.shape[2];
    return [ this.keys, this.values ];
  }

  private updateInPlace(keys: mx.array, values: mx.array): [ mx.array, mx.array ] {
    // May not have hit the max size yet, so potentially keep growing the cache.
    const [ B, nKVHeads, S, kHeadDim ] = keys.shape;
    const prev = this.#offset;
    if (!this.keys || (prev >= this.keys.shape[2] && this.keys.shape[2] < this.maxSize)) {
      const vHeadDim = values.shape[3];
      const newSize = Math.min(this.step, this.maxSize - prev);
      const kShape = [ B, nKVHeads, newSize, kHeadDim ];
      const vShape = [ B, nKVHeads, newSize, vHeadDim ];
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
    const insert: mx.ArrayIndex[] = [ '...', mx.Slice(this.#idx, this.#idx + S), mx.Slice() ];
    this.keys.indexPut_(insert, keys);
    this.values.indexPut_(insert, values);
    this.#offset += S;
    this.#idx += S;

    // If the buffer is not full, slice off the end.
    if (this.#offset < this.maxSize) {
      const get: mx.ArrayIndex[] = [ '...', mx.Slice(null, this.#offset), mx.Slice() ];
      return [ this.keys.index(...get), this.values.index(...get) ];
    }
    return [ this.keys, this.values ];
  }

  private trim(trimSize: number, v: mx.array, append?: mx.array) {
    let toCat: mx.array[];
    if (trimSize > 0) {
      toCat = [ v.index('...', mx.Slice(null, this.keep), mx.Slice()),
                v.index('...', mx.Slice(trimSize + this.keep), mx.Slice()) ];
    } else {
      toCat = [ v ];
    }
    if (append) {
      toCat.push(append);
    }
    return mx.concatenate(toCat, 2);
  }

  // Rearrange the cache into temporal order, slicing off the end if unused.
  private temporalOrder(v: mx.array) {
    if (this.#idx == v.shape[2]) {
      return v;
    } else if (this.#idx < this.#offset) {
      return mx.concatenate([
        v.index('...', mx.Slice(null, this.keep), mx.Slice()),
        v.index('...', mx.Slice(this.#idx), mx.Slice()),
        v.index('...', mx.Slice(this.keep, this.#idx), mx.Slice()),
      ], 2);
    } else {
      return v.index('...', mx.Slice(null, this.#idx), mx.Slice());
    }
  }
}
