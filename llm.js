import fs from 'node:fs/promises'
import path from 'node:path'
import nextTick from 'tick-promise'
import {TokenizerLoader} from '@lenml/tokenizers'
import {core as mx, nn} from '@frost-beta/mlx'

// A design of KV cache friendly to MLX's memory cache design, which allocates
// arrays in same shapes.
// See also https://github.com/ml-explore/mlx-examples/issues/724.
export class KVCache {
  constructor(headDim, nKVHeads) {
    this.nKVHeads = nKVHeads
    this.headDim = headDim
    this.keys = null
    this.values = null
    this.offset = 0
    this.step = 256
  }

  updateAndFetch(keys, values) {
    const prev = this.offset
    if (!this.keys || (prev + keys.shape[2] > this.keys.shape[2])) {
      const nSteps = Math.floor((this.step + keys.shape[2] - 1) / this.step)
      const shape = [1, this.nKVHeads, nSteps * this.step, this.headDim]
      const newK = mx.zeros(shape, keys.dtype)
      const newV = mx.zeros(shape, values.dtype)
      if (this.keys) {
        const old = [this.keys, this.values]
        if (prev % this.step != 0) {
          const get = ['...', mx.Slice(null, prev), mx.Slice()]
          this.keys = this.keys.index(get)
          this.values = this.values.index(get)
        }
        this.keys = mx.concatenate([this.keys, newK], 2)
        this.values = mx.concatenate([this.values, newV], 2)
        mx.dispose(old)
      } else {
        this.keys = newK
        this.values = newV
      }
    }

    this.offset += keys.shape[2]

    const insert = ['...', mx.Slice(prev, this.offset), mx.Slice()]
    this.keys.indexPut_(insert, keys)
    this.values.indexPut_(insert, values)

    const get = ['...', mx.Slice(null, this.offset), mx.Slice()]
    return [this.keys.index(...get), this.values.index(...get)]
  }
}

// Convert snake_case args into camelCase args.
export function baseModelArgs(args) {
  const newArgs = {}
  for (const key in args) {
    const newKey = key.replace(/(\_\w)/g, (s) => s[1].toUpperCase())
    newArgs[newKey] = args[key]
  }
  return newArgs
}

// Create an additive causal mask.
export function createAdditiveCausalMask(N, offset = 0) {
  const rinds = mx.arange(offset + N)
  const linds = offset ? mx.arange(offset, offset + N) : rinds
  const mask = mx.less(linds.index(mx.Slice(), null), rinds.index(null))
  return mx.multiply(mask, -1e9)
}

// Return a tokenizer.
export async function loadTokenizer(dir) {
  return TokenizerLoader.fromPreTrained({
    tokenizerJSON: JSON.parse(await fs.readFile(path.join(dir, 'tokenizer.json'))),
    tokenizerConfig: JSON.parse(await fs.readFile(path.join(dir, 'tokenizer_config.json'))),
  })
}

// Return a model.
export async function loadModel(dir) {
  // Read model config and weights.
  const config = JSON.parse(await fs.readFile(path.join(dir, 'config.json')))
  const weights = {}
  for (const filename of await fs.readdir(dir)) {
    if (filename.endsWith('.safetensors'))
      Object.assign(weights, mx.load(path.join(dir, filename)))
  }

  // Create llama3 model.
  let model
  try {
    const {Model} = await import(`./models/${config.model_type}.js`)
    model = new Model(config)
  } catch (error) {
    if (error.code == 'ERR_MODULE_NOT_FOUND') {
      console.error('Unsupported model type:', config.model_type)
      process.exit(1)
    }
    throw error
  }

  // Quantization.
  if (config.quantization) {
    const predicate = (p, m) => {
      // Some legacy models which may not have everything quantized.
      return (`${p}.scales` in weights) &&
             ((m instanceof nn.Linear) || (m instanceof nn.Embedding))
    }
    const {group_size: groupSize, bits} = config.quantization
    nn.quantize(model, groupSize, bits, predicate)
  }

  // Load weights.
  model.loadWeights(Object.entries(weights))
  mx.eval(model.parameters())
  return model
}

// Generate tokens from prompt.
export async function* step(promptTokens, model, eosToken, topP = 1, temperature = 1) {
  // Create KV Cache.
  const cache = []
  for (let i = 0; i < model.layers.length; ++i)
    cache[i] = new KVCache(model.headDim, model.nKVHeads)

  // Feed the tokens to model and get predictions.
  const forward = (y) => {
    let logits = model.forward(mx.array([y], mx.int32), cache)
    logits = logits.index(mx.Slice(), -1, mx.Slice())
    const [token, prob] = sample(logits, topP, temperature)
    // The cache is also returned so it does not get freed by mx.tidy().
    return [token.item(), prob.item(), cache]
  }

  let tokens = promptTokens
  while (true) {
    // Forward the tokens to model, and make sure intermediate tensors are freed.
    const [token, prob] = mx.tidy(() => forward(tokens))
    // Quit after getting EOS.
    if (token == eosToken)
      break
    tokens = [token]
    // Yield the result in the next tick of loop, so GC can get a chance to run.
    await nextTick()
    yield [token, prob]
  }

  // Make sure cache is cleared after generation is done.
  mx.dispose(cache)
}

// Pick the best token from logits.
export function sample(logits, topP, temperature) {
  const softmaxLogits = mx.softmax(logits)
  let token
  if (temperature === 0) {
    token = mx.argmax(logits, -1)
  } else {
    if (topP > 0 && topP < 1) {
      token = topPSampling(logits, topP, temperature)
    } else {
      token = mx.random.categorical(mx.multiply(logits, 1 / temperature))
    }
  }
  const prob = softmaxLogits.index(0, token)
  return [token, prob]
}

// Sampling with top-p.
export function topPSampling(logits, topP, temperature) {
  const probs = mx.softmax(mx.divide(logits, temperature), -1)

  const sortedIndices = mx.argsort(probs, -1)
  const sortedProbs = probs.index('...', sortedIndices.squeeze(0))

  const cumulativeProbs = mx.cumsum(sortedProbs, -1)

  const topProbs = mx.where(mx.greater(cumulativeProbs, mx.subtract(1, topP)),
                            sortedProbs,
                            mx.zerosLike(sortedProbs))

  const sortedToken = mx.random.categorical(mx.log(topProbs))
  return sortedIndices.squeeze(0).index(sortedToken)
}
