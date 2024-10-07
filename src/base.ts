import {core as mx, nn} from '@frost-beta/mlx';
import {KVCacheOptions, BaseKVCache, RotatingKVCache} from './kv-cache.js';
import {loadWeights, readJsonSync} from './fs.js';

/**
 * The base class of LLM models with or without vision.
 */
export abstract class BaseModel extends nn.Module {
  /**
   * Whether this model has an encoder.
   */
  hasEncoder = false;
  /**
   * The token that used as the first token for decoder.
   */
  decoderStartToken?: number;
  /**
   * The special token representing image, usually <image> or <|image|>.
   */
  imagePlaceholder?: string;
  /**
   * The token id of the placeholder.
   */
  imageToken?: number;

  /**
   * Return vision embeddings for the preprocessed images.
   */
  computePixelEmbeddings(pixels: mx.array): mx.array {
    throw new Error('This model does not have vision.');
  }

  /**
   * Return text embeddings for the encoded text.
   */
  abstract computeTextEmbeddings(inputs: mx.array): mx.array;

  /**
   * Predict next token for the embeddings.
   */
  abstract decodeEmbeddings(embeddings: mx.array, memory?: mx.array, cache?: BaseKVCache[]): mx.array;

  /**
   * Pass the text embeddings to encoder and return memory.
   */
  encodeEmbeddings(embeddings: mx.array, cache?: BaseKVCache[]): mx.array {
    throw new Error('This model has no encoder.');
  }

  /**
   * Predict next token for the encoded text.
   */
  override forward(inputs: mx.array, memory?: mx.array, cache?: BaseKVCache[]): mx.array {
    return this.decodeEmbeddings(this.computeTextEmbeddings(inputs), memory, cache);
  }

  /**
   * Modify the model weights for MLX models.
   */
  sanitize(weights: Record<string, mx.array>) {
  }

  /**
   * Replace with image tokens in text embeddings with image embeddings.
   */
  mergeTextPixelEmbeddings(inputs: mx.array, inputEmbeds: mx.array, pixelEmbeds: mx.array) {
    if (inputs.shape[0] != 1)
      throw Error('Only implemented with batch size of 1.');

    // Find out the indices of <image> placeholders in encoded text.
    const imagePositions: number[] = [];
    const tokens = inputs.index(0).tolist() as number[];
    for (let i = 0; i < tokens.length; ++i) {
      if (tokens[i] == this.imageToken)
        imagePositions.push(i);
    }

    // The number of <image> placeholders should match pixelEmbeds size.
    const nImages = pixelEmbeds.shape[0];
    if (imagePositions.length != nImages) {
      throw new Error(`The number of image tokens (${imagePositions.length}) does not match the number of image inputs (${nImages}).`);
    }

    // Split the inputs by <image>.
    const inputEmbedsSegments: mx.array[] = [];
    let startIdx = 0;
    for (const position of imagePositions) {
      inputEmbedsSegments.push(inputEmbeds.index(mx.Slice(), mx.Slice(startIdx, position)));
      startIdx = position + 1;
    }

    // Concatenate them together.
    const segments: mx.array[] = [];
    for (let i = 0; i < inputEmbedsSegments.length; ++i) {
      segments.push(inputEmbedsSegments[i], pixelEmbeds.index(i).index(mx.newaxis));
    }
    segments.push(inputEmbeds.index(mx.Slice(), mx.Slice(startIdx)));
    return mx.concatenate(segments, 1);
  }

  // Following methods are defined for internal KV cache use only.
  abstract getDecoderKVCacheOptions(): KVCacheOptions;
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
  const mask = mx.less(linds.index(mx.Slice(), mx.newaxis),
                       rinds.index(mx.newaxis));
  return mx.multiply(mask, -1e9);
}

/**
 * Create an attention mask.
 */
export function createAttentionMask(h: mx.array, cache?: BaseKVCache[]) {
  const T = h.shape[1];
  if (T > 1) {
    let offset: number;
    if (cache) {
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
  return model;
}

/**
 * Options passed to step.
 */
export interface StepOptions {
  signal?: AbortSignal;
  kvCache?: BaseKVCache[];
  topP?: number;
  temperature?: number;
}

/**
 * Generate tokens from prompt.
 */
export async function* step(promptEmbeds: mx.array,
                            model: BaseModel,
                            eosToken: number,
                            {
                              signal,
                              kvCache,
                              topP = 0.8,
                              temperature = 1,
                            }: StepOptions = {}): AsyncGenerator<number, void> {
  // Create KV Cache if none is specified in options.
  const cache = kvCache ?? RotatingKVCache.create(model.getDecoderKVCacheOptions());

  // Sample the logits results.
  const predict = async (logits: mx.array) => {
    logits = logits.index(mx.Slice(), -1);
    const [ token ] = sample(logits, topP, temperature);
    await mx.asyncEval(token);
    return token.item() as number;
  };

  // Handle prompt: for encoder-decoder models we pass it to encoder, fo
  // decoder-only models we pass it to decoder directly.
  let nextToken: number;
  let memory: mx.array | undefined;
  if (model.hasEncoder) {
    nextToken = model.decoderStartToken;
    memory = model.encodeEmbeddings(promptEmbeds);
  } else {
    // Forward prompt by steps so we don't use too much RAM.
    // See also https://github.com/ml-explore/mlx-examples/pull/931
    const prefillStepSize = 512;
    const embeddingsSize = promptEmbeds.shape[1];
    for (let offset = 0; offset < embeddingsSize;) {
      if (signal?.aborted)
        break;
      await mx.tidy(async () => {
        const size = Math.min(prefillStepSize, embeddingsSize - offset);
        const chunk = promptEmbeds.index(mx.Slice(), mx.Slice(offset, offset + size));
        const logits = model.decodeEmbeddings(chunk, undefined, cache);
        mx.eval(cache.map(c => c.state));
        offset += size;
        // Do token-by-token generation after prompt is consumed.
        if (offset == embeddingsSize)
          nextToken = await predict(logits);
        // Keep the cache from being released.
        return cache;
      });
    }
  }

  do {
    // Quit after getting EOS.
    if (nextToken == eosToken)
      break;
    // The generation is aborted.
    if (signal?.aborted)
      break;
    // Do not yield token if it is the decoderStartToken.
    if (!(model.hasEncoder && nextToken == model.decoderStartToken))
      yield nextToken;
    // Forward the token to model and free intermediate tensors.
    [ nextToken ] = await mx.tidy(async () => {
      const logits = model.forward(mx.array([ [ nextToken ] ], mx.int32), memory, cache);
      // The cache is also returned so it does not get freed by mx.tidy().
      return [ await predict(logits), cache ];
    });
  } while (true);

  // Make sure the temporary cache is cleared after generation is done.
  if (!kvCache) {
    mx.dispose(cache);
  }
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
