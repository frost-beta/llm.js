import {core as mx, nn} from '@frost-beta/mlx'
import {baseModelArgs, createAttentionMask} from '../llm.js'

function modelArgs(args) {
  args = Object.assign({
    attentionBias: false,
    mlpBias: false,
    ropeTheta: 10000,
    ropeTraditional: false,
    tieWordEmbeddings: true,
  }, baseModelArgs(args))
  if (!args.numKeyValueHeads) {
    args.numKeyValueHeads = args.numAttentionHeads
  }
  if (args.ropeScaling) {
    if (!args.ropeScaling.factor)
      throw new Error('rope_scaling must contain "factor"')
    const ropeType = args.ropeScaling.type || args.ropeScaling.ropeType
    if (!ropeType)
      throw new Error('rope_scaling must contain either "type" or "rope_type"')
    if (!['linear', 'dynamic', 'llama3'].includes(ropeType))
      throw new Error('rope_scaling "type" currently only supports "linear", "dynamic" or "llama3"')
  }
  return args
}

class DynamicNTKScalingRoPE extends nn.Module {
  constructor(dims,
              maxPositionEmbeddings = 2048,
              traditional = false,
              base = 10000,
              scale = 1.0,
              ropeType = 'default',
              ropeScaling = null) {
    super()
    this.dims = dims
    this.maxPositionEmbeddings = maxPositionEmbeddings
    this.traditional = traditional
    this.originalBase = base
    this.scale = scale
    this.ropeType = ropeType
    this.ropeScaling = ropeScaling
    this.base = this.computeBaseFreq()
  }

  computeBaseFreq() {
    if (this.ropeType === 'llama3')
      return this.computeLlama3BaseFreq()
    return this.originalBase
  }

  computeLlama3BaseFreq() {
    const factor = this.ropeScaling.factor
    const lowFreqFactor = this.ropeScaling.lowFreqFactor ?? 1.0
    const highFreqFactor = this.ropeScaling.highFreqFactor ?? 4.0
    const oldContextLen = this.ropeScaling.originalMaxPositionEmbeddings ?? 8192

    const lowFreqWavelen = oldContextLen / lowFreqFactor
    const highFreqWavelen = oldContextLen / highFreqFactor

    const freqs = mx.power(this.originalBase, mx.divide(mx.arange(0, this.dims, 2), this.dims))
    const wavelens = mx.multiply(2 * mx.pi, freqs)

    const smooths = mx.divide(mx.subtract(wavelens, highFreqWavelen),
                              mx.subtract(lowFreqWavelen, highFreqWavelen))
    let newBaseFreqs = mx.add(mx.multiply(mx.multiply(freqs,
                                                      mx.subtract(1, smooths)),
                                          factor),
                              smooths);
    newBaseFreqs = mx.where(mx.less(wavelens, highFreqWavelen), freqs, newBaseFreqs)
    newBaseFreqs = mx.where(mx.greater(wavelens, lowFreqWavelen), mx.multiply(freqs, factor), newBaseFreqs)
    return mx.mean(newBaseFreqs).item()
  }

  forward(x, offset = 0) {
    const seqLen = x.shape[1] + offset
    let base = this.base
    if (this.maxPositionEmbeddings && seqLen > this.maxPositionEmbeddings) {
      base *= ((this.scale * seqLen / this.maxPositionEmbeddings) - (this.scale - 1)) ** (this.dims / (this.dims - 2))
    }
    return mx.fast.rope(x, this.dims, this.traditional, base, this.scale, offset)
  }
}

function initializeRoPE(args) {
  const headDim = args.headDim ?? Math.floor(args.hiddenSize / args.numAttentionHeads)

  const ropeScaling = args.ropeScaling
  let ropeType = 'default'
  let ropeScale = 1.0

  if (ropeScaling) {
    ropeType = (ropeScaling.type ?? ropeScaling.ropeType) ?? 'default'
    if (ropeType === 'linear')
      ropeScale = 1 / ropeScaling.factor
    else if (ropeType === 'llama3')
      ropeScale = 1.0
  }

  return new DynamicNTKScalingRoPE(headDim,
                                   args.maxPositionEmbeddings,
                                   args.ropeTraditional,
                                   args.ropeTheta,
                                   ropeScale,
                                   ropeType,
                                   ropeScaling)
}

class Attention extends nn.Module {
  constructor(args) {
    super()
    const dim = args.hiddenSize
    this.nHeads = args.numAttentionHeads
    this.nKVHeads = args.numKeyValueHeads

    const headDim = Math.floor(args.hiddenSize / this.nHeads)
    this.scale = headDim ** -0.5

    this.qProj = new nn.Linear(dim, this.nHeads * headDim, args.attentionBias)
    this.kProj = new nn.Linear(dim, this.nKVHeads * headDim, args.attentionBias)
    this.vProj = new nn.Linear(dim, this.nKVHeads * headDim, args.attentionBias)
    this.oProj = new nn.Linear(this.nHeads * headDim, dim, args.attentionBias)

    this.rope = initializeRoPE(args);
  }

  forward(x, mask, cache) {
    const [B, L, D] = x.shape

    let queries = this.qProj.forward(x)
    let keys = this.kProj.forward(x)
    let values = this.vProj.forward(x)

    // Prepare the queries, keys and values for the attention computation.
    queries = queries.reshape(B, L, this.nHeads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, this.nKVHeads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, this.nKVHeads, -1).transpose(0, 2, 1, 3)

    if (cache) {
      queries = this.rope.forward(queries, cache.offset)
      keys = this.rope.forward(keys, cache.offset);
      [keys, values] = cache.updateAndFetch(keys, values)
    } else {
      queries = this.rope.forward(queries)
      keys = this.rope.forward(keys)
    }

    let output = mx.fast.scaledDotProductAttention(queries, keys, values, this.scale, mask)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return this.oProj.forward(output)
  }
}

class MLP extends nn.Module {
  constructor(args) {
    super()

    const dim = args.hiddenSize
    const hiddenDim = args.intermediateSize
    const mlpBias = args.mlpBias

    this.gateProj = new nn.Linear(dim, hiddenDim, mlpBias)
    this.downProj = new nn.Linear(hiddenDim, dim, mlpBias)
    this.upProj = new nn.Linear(dim, hiddenDim, mlpBias)
  }

  forward(x) {
    return this.downProj.forward(mx.multiply(nn.silu(this.gateProj.forward(x)),
                                             this.upProj.forward(x)))
  }
}

class TransformerBlock extends nn.Module {
  constructor(args) {
    super()
    this.numAttentionHeads = args.numAttentionHeads
    this.hiddenSize = args.hiddenSize
    this.selfAttn = new Attention(args)
    this.mlp = new MLP(args)
    this.inputLayernorm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps)
    this.postAttentionLayernorm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps)
  }

  forward(x, mask, cache) {
    const r = this.selfAttn.forward(this.inputLayernorm.forward(x), mask, cache)
    const h = mx.add(x, r)
    return mx.add(h, this.mlp.forward(this.postAttentionLayernorm.forward(h)))
  }
}

class LlamaModel extends nn.Module {
  constructor(args) {
    super()
    this.vocabSize = args.vocabSize
    this.numHiddenLayers = args.numHiddenLayers
    this.embedTokens = new nn.Embedding(args.vocabSize, args.hiddenSize)
    this.layers = []
    for (let i = 0; i < args.numHiddenLayers; ++i)
      this.layers.push(new TransformerBlock(args))
    this.norm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps)
  }

  forward(inputs, cache) {
    let h = this.embedTokens.forward(inputs)

    const mask = createAttentionMask(h, cache);

    cache = cache ?? new Array(this.layers.length)

    for (let i in this.layers)
      h = this.layers[i].forward(h, mask, cache[i])

    return this.norm.forward(h)
  }
}

export class Model extends nn.Module {
  constructor(obj) {
    const args = modelArgs(obj)
    super()

    this.args = args
    this.modelType = args.modelType
    this.model = new LlamaModel(args)
    if (!args.tieWordEmbeddings)
      this.lmHead = new nn.Linear(args.hiddenSize, args.vocabSize, false)
  }

  forward(inputs, cache) {
    const out = this.model.forward(inputs, cache)
    if (this.args.tieWordEmbeddings)
      return this.model.embedTokens.asLinear(out)
    else
      return this.lmHead.forward(out)
  }

  get layers() {
    return this.model.layers
  }

  get headDim() {
    return Math.floor(this.args.hiddenSize / this.args.numAttentionHeads)
  }

  get nKVHeads() {
    return this.args.numKeyValueHeads
  }
}
