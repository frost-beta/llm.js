import {core as mx, nn} from '@frost-beta/mlx'
import {KVCache, baseModelArgs, createAdditiveCausalMask} from '../llm.js'

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
    const requiredKeys = [ 'factor', 'type' ]
    if (!Object.keys(args.ropeScaling).every(key => requiredKeys.includes(key)))
      throw Error(`rope_scaling must contain keys ${requiredKeys}`)
    if (this.ropeScaling.type != 'linear')
      throw Error("rope_scaling 'type' currently only supports 'linear'")
  }
  return args
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

    const ropeScale = args.ropeScaling?.type == 'linear' ? 1 / args.ropeScaling.factor
                                                         : 1
    this.rope = new nn.RoPE(headDim, args.ropeTraditional, args.ropeTheta, ropeScale)
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

    let mask
    if (h.shape[1] > 1) {
      mask = createAdditiveCausalMask(h.shape[1], cache ? cache[0].offset : 0)
      mask = mask.astype(h.dtype)
    }

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
      return this.model.embedTokens.asLinear.forward(out)
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
