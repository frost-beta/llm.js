import {core as mx, nn} from '@frost-beta/mlx'
import {baseModelArgs} from '../gemma.js'

function modelArgs(args) {
  return Object.assign({
    ropeTheta: 1000000,
    ropeTraditional: false,
  }, baseModelArgs(args))
}

class RMSNorm extends nn.Module {
  constructor(dim, eps = 1e-5) {
    super()
    this.weight = mx.ones([dim])
    this.eps = eps
  }

  forward(x) {
    return mx.fast.rmsNorm(x, mx.add(1.0, this.weight), this.eps)
  }
}

class Attention extends nn.Module {
  constructor(args) {
    super()
    const dim = args.hiddenSize
    this.nHeads = args.numAttentionHeads
    this.nKVHeads = args.numKeyValueHeads
    this.headDim = args.headDim

    this.scale = this.headDim ** -0.5

    this.qProj = new nn.Linear(dim, this.nHeads * this.headDim, false)
    this.kProj = new nn.Linear(dim, this.nKVHeads * this.headDim, false)
    this.vProj = new nn.Linear(dim, this.nKVHeads * this.headDim, false)
    this.oProj = new nn.Linear(this.nHeads * this.headDim, dim, false)

    this.rope = new nn.RoPE(this.headDim, args.ropeTraditional, args.ropeTheta)
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
  constructor(dim, hiddenDim) {
    super()
    this.gateProj = new nn.Linear(dim, hiddenDim, false)
    this.downProj = new nn.Linear(hiddenDim, dim, false)
    this.upProj = new nn.Linear(dim, hiddenDim, false)
  }

  forward(x) {
    return this.downProj.forward(mx.multiply(nn.gelu(this.gateProj.forward(x)),
                                             this.upProj.forward(x)))
  }
}

class TransformerBlock extends nn.Module {
  constructor(args) {
    super()
    this.numAttentionHeads = args.numAttentionHeads
    this.hiddenSize = args.hiddenSize
    this.selfAttn = new Attention(args)
    this.mlp = new MLP(args.hiddenSize, args.intermediateSize)
    this.inputLayernorm = new RMSNorm(args.hiddenSize, args.rmsNormEps)
    this.postAttentionLayernorm = new RMSNorm(args.hiddenSize, args.rmsNormEps)
  }

  forward(x, mask, cache) {
    const r = this.selfAttn.forward(this.inputLayernorm.forward(x), mask, cache)
    const h = mx.add(x, r)
    return mx.add(h, this.mlp.forward(this.postAttentionLayernorm.forward(h)))
  }
}

class GemmaModel extends nn.Module {
  constructor(args) {
    super()
    this.vocabSize = args.vocabSize
    this.numHiddenLayers = args.numHiddenLayers
    this.embedTokens = new nn.Embedding(args.vocabSize, args.hiddenSize)
    this.layers = []
    for (let i = 0; i < args.numHiddenLayers; ++i)
      this.layers.push(new TransformerBlock(args))
    this.norm = new RMSNorm(args.hiddenSize, args.rmsNormEps)
  }

  forward(inputs, cache) {
    let h = this.embedTokens.forward(inputs)
    h = mx.multiply(h, this.hiddenSize ** 0.5)

    let mask
    if (h.shape[1] > 1) {
      mask = nn.MultiHeadAttention.createAdditiveCausalMask(h.shape[1])
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

    this.modelType = args.modelType
    this.model = new GemmaModel(args)
    this.args = args
  }

  forward(inputs, cache) {
    const out = this.model.forward(inputs, cache)
    return this.model.embedTokens.asLinear(out)
  }

  get layers() {
    return this.model.layers
  }

  get headDim() {
    return this.args.headDim
  }

  get nKVHeads() {
    return this.args.numKeyValueHeads
  }
}
