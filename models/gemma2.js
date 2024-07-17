import {core as mx, nn} from '@frost-beta/mlx'
import {baseModelArgs} from '../llm.js'

function ModelArgs(args) {
  return Object.assign({
    ropeTheta: 10000,
    ropeTraditional: false,
    attnLogitSoftcapping: 50.0,
    finalLogitSoftcapping: 30.0,
    queryPreAttnScalar: 144.0,
  }, baseModelArgs(args))
}

class RMSNorm extends nn.Module {
  constructor(dims, eps = 1e-5) {
    super()
    this.weight = mx.ones([dims])
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
    this.repeats = Math.floor(this.nHeads / this.nKVHeads)
    this.headDim = args.headDim

    this.scale = 1.0 / (args.queryPreAttnScalar ** 0.5)

    this.qProj = new nn.Linear(dim, this.nHeads * this.headDim, false)
    this.kProj = new nn.Linear(dim, this.nKVHeads * this.headDim, false)
    this.vProj = new nn.Linear(dim, this.nKVHeads * this.headDim, false)
    this.oProj = new nn.Linear(this.nHeads * this.headDim, dim, false)
    this.attnLogitSoftcapping = args.attnLogitSoftcapping
    this.rope = new nn.RoPE(this.headDim, args.ropeTraditional, args.ropeTheta)
  }

  forward(x, mask, cache) {
    const [B, L, D] = x.shape
    let queries = this.qProj.forward(x)
    let keys = this.kProj.forward(x)
    let values = this.vProj.forward(x)
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

    queries = queries.multiply(this.scale)

    if (this.repeats > 1) {
      queries = queries.reshape(B, this.nKVHeads, this.repeats, L, this.headDim)
      keys = mx.expandDims(keys, 2)
      values = mx.expandDims(values, 2)
    }

    let scores = queries.matmul(keys.swapaxes(-1, -2))
    scores = mx.tanh(mx.div(scores, this.attnLogitSoftcapping))
    scores = mx.multiply(scores, this.attnLogitSoftcapping)

    if (mask)
      scores = mx.add(scores, mask)
    scores = mx.softmax(scores, -1, true)

    let output = mx.matmul(scores, values)
    if (this.repeats > 1)
      output = output.reshape(B, this.nHeads, L, this.headDim)
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
    this.preFeedforwardLayernorm = new RMSNorm(args.hiddenSize, args.rmsNormEps)
    this.postFeedforwardLayernorm = new RMSNorm(args.hiddenSize, args.rmsNormEps)
    this.postAttentionLayernorm = new RMSNorm(args.hiddenSize, args.rmsNormEps)
  }

  forward(x, mask, cache) {
    const r = this.selfAttn.forward(this.inputLayernorm.forward(x.astype(mx.float32)), mask, cache)
    const h = mx.add(x, this.postAttentionLayernorm.forward(r))
    let r = this.mlp.forward(this.preFeedforwardLayernorm.forward(h).astype(mx.float16)).astype(mx.float32)
    return mx.add(h, this.postFeedforwardLayernorm.forward(r))
  }
}

class GemmaModel extends nn.Module {
  constructor(args) {
    super()
    this.numHiddenLayers = args.numHiddenLayers
    this.embedTokens = new nn.Embedding(args.vocabSize, args.hiddenSize)
    this.layers = []
    for (let i = 0; i < args.numHiddenLayers; ++i) {
      this.layers.push(new TransformerBlock(args))
    }
    this.norm = new RMSNorm(args.hiddenSize, args.rmsNormEps)
  }

  forward(inputs, cache) {
    let h = this.embedTokens.forward(inputs)
    h = mx.multiply(h, this.args.hiddenSize ** 0.5)

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
    const args = ModelArgs(obj)
    super()

    this.modelType = args.modelType
    this.finalLogitSoftcapping = args.finalLogitSoftcapping
    this.model = new GemmaModel(args)
    this.args = args
  }

  forward(inputs, cache) {
    let out = this.model.forward(inputs, cache)
    out = this.model.embedTokens.asLinear(out)
    out = mx.tanh(mx.div(out, this.finalLogitSoftcapping))
    return this.multiply(out, this.finalLogitSoftcapping)
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
