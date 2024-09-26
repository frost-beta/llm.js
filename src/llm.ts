import {core as mx, nn} from '@frost-beta/mlx';
import {loadWeights, readJsonSync} from './fs.js';

export * from './image-processor.js';
export * from './tokenizer.js';

/**
 * The base class of LLM models.
 */
export abstract class BaseModel extends nn.Module {
  abstract forward(inputs: mx.array, cache?: BaseKVCache[]): mx.array;

  forwardWithPixels(inputIds: mx.array, pixelValues: mx.array, cache?: BaseKVCache[]): mx.array {
    return this.forward(inputIds, cache);
  }

  sanitize(weights: Record<string, mx.array>) {
  }

  abstract get layers(): nn.Module[];
  abstract get headDim(): number;
  abstract get nKVHeads(): number;
}

/**
 * The base class of KV cache.
 */
export abstract class BaseKVCache {
  keys?: mx.array;
  values?: mx.array;
  offset = 0;
  step = 256;

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

/**
 * Convert snake_case args into camelCase args.
 */
export function baseModelArgs<T>(args: T): T {
  if (Array.isArray(args))
    return args.map(v => baseModelArgs(v)) as T;
  if (typeof args != 'object' || args === null)
    return args;
  const newArgs: Record<string, any> = {};
  for (const key in args) {
    const newKey = key.replace(/(\_\w)/g, (s) => s[1].toUpperCase())
    newArgs[newKey] = baseModelArgs(args[key]);
  }
  return newArgs as T;
}

/**
 * Create an additive causal mask.
 */
export function createAdditiveCausalMask(N: number, offset = 0) {
  const rinds = mx.arange(offset + N);
  const linds = offset ? mx.arange(offset, offset + N) : rinds;
  const mask = mx.less(linds.index(mx.Slice(), mx.newaxis), rinds.index(mx.newaxis));
  return mx.multiply(mask, -1e9);
}

/**
 * Create an attention mask.
 */
export function createAttentionMask(h: mx.array, cache?: BaseKVCache[]) {
  const T = h.shape[1];
  if (T > 1) {
    let offset: number;
    if (cache && cache[0]) {
      const c = cache[0];
      if (c instanceof RotatingKVCache)
        offset = Math.min(c.maxSize - 1, c.offset);
      else
        offset = c.offset;
    } else {
      offset = 0;
    }
    return createAdditiveCausalMask(T, offset).astype(h.dtype);
  } else {
    return null;
  }
}

/**
 * Load the model from directory.
 */
export async function loadModel(dir: string): Promise<BaseModel> {
  // Create llama3 model.
  const config = readJsonSync(`${dir}/config.json`);
  let model: BaseModel;
  try {
    const {Model} = await import(`./models/${config.model_type}.js`);
    model = new Model(config);
  } catch (error) {
    if (error.code == 'ERR_MODULE_NOT_FOUND') {
      console.error('Unsupported model type:', config.model_type);
      process.exit(1);
    }
    throw error;
  }

  // Read and sanitize weights.
  const weights = loadWeights(dir);
  model.sanitize(weights);

  // Quantization.
  if (config.quantization) {
    const predicate = (p: string, m: nn.Module) => {
      // Some legacy models which may not have everything quantized.
      return (`${p}.scales` in weights) &&
             ((m instanceof nn.Linear) || (m instanceof nn.Embedding))
    }
    const {groupSize, bits} = config.quantization;
    nn.quantize(model, groupSize, bits, predicate);
  }

  // Load weights.
  model.loadWeights(Object.entries(weights));
  mx.eval(model.parameters());
  return model;
}

/**
 * Generate tokens from prompt.
 */
export async function* step(promptTokens: number[],
                            model: BaseModel,
                            eosToken: number,
                            topP = 1,
                            temperature = 1): AsyncGenerator<[ number, number ], void> {
  // Create KV Cache.
  const cache: BaseKVCache[] = [];
  for (let i = 0; i < model.layers.length; ++i)
    cache[i] = new RotatingKVCache(model.headDim, model.nKVHeads);

  // Feed the tokens to model and get predictions.
  const forward = (y: number[]): [ number, number, BaseKVCache[] ] => {
    let logits = model.forward(mx.array([ y ], mx.int32), cache);
    logits = logits.index(mx.Slice(), -1, mx.Slice());
    const [ token, prob ] = sample(logits, topP, temperature);
    // The cache is also returned so it does not get freed by mx.tidy().
    return [ token.item() as number, prob.item() as number, cache ];
  }

  // Prefill the prompt tokens.
  // See also https://github.com/ml-explore/mlx-examples/pull/931
  const prefillStepSize = 512;
  let tokens = promptTokens;
  while (tokens.length > prefillStepSize) {
    mx.tidy(() => {
      model.forward(mx.array(tokens.slice(0, prefillStepSize), mx.int32).index(mx.newaxis), cache);
      mx.eval(cache.map(c => c.state));
      tokens = tokens.slice(prefillStepSize);
      return cache;
    });
  }

  while (true) {
    // Forward the tokens to model, and make sure intermediate tensors are freed.
    const [ token, prob ] = mx.tidy(() => forward(tokens));
    // Quit after getting EOS.
    if (token == eosToken)
      break;
    tokens = [ token ];
    // Yield the result in the next tick of loop, so GC can get a chance to run.
    yield await new Promise(resolve => {
      process.nextTick(() => resolve([ token, prob ]));
    });
  }

  // Make sure cache is cleared after generation is done.
  mx.dispose(cache);
}

/**
 * Pick the best token from logits.
 */
export function sample(logits: mx.array, topP = 1, temperature = 1): [ mx.array, mx.array ] {
  const softmaxLogits = mx.softmax(logits);
  let token: mx.array;
  if (temperature === 0) {
    token = mx.argmax(logits, -1);
  } else {
    if (topP > 0 && topP < 1) {
      token = topPSampling(logits, topP, temperature);
    } else {
      token = mx.random.categorical(mx.multiply(logits, 1 / temperature));
    }
  }
  const prob = softmaxLogits.index(0, token);
  return [ token, prob ];
}

/**
 * Sampling with top-p.
 */
export function topPSampling(logits: mx.array, topP = 1, temperature = 1): mx.array {
  const probs = mx.softmax(mx.divide(logits, temperature), -1);

  const sortedIndices = mx.argsort(probs, -1);
  const sortedProbs = probs.index('...', sortedIndices.squeeze(0));

  const cumulativeProbs = mx.cumsum(sortedProbs, -1);

  const topProbs = mx.where(mx.greater(cumulativeProbs, mx.subtract(1, topP)),
                            sortedProbs,
                            mx.zerosLike(sortedProbs));

  const sortedToken = mx.random.categorical(mx.log(topProbs));
  return sortedIndices.squeeze(0).index(sortedToken);
}
