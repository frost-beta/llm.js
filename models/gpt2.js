import {core as mx, nn} from '@frost-beta/mlx'
import {baseModelArgs, createAdditiveCausalMask} from '../gpt2.js'

function modelArgs(args) {
  args = baseModelArgs(args)
  if (!args.numKeyValueHeads)
    args.numKeyValueHeads = args.nHead
  return args
}

class Attention extends nn.Module {
  constructor(args) {
    super()

    this.nEmbd = args.nEmbd
    this.nHead = args.nHead
    if (args.nEmbd % args.nHead != 0)
      throw Error('nEmbd must be divisible by nHead')
    this.headDim = this.nEmbd / this.nHead

    this.scale = this.headDim ** -0.5

    this.cAttn = new nn.Linear(this.nEmbd, 3 * this.nEmbd, true)
    this.cProj = new nn.Linear(this.nEmbd, this.nEmbd, true)
  }

  forward(x, mask, cache) {
    const [B, L, D] = x.shape

    const qkv = this.cAttn.forward(x)
    let [queries, keys, values] = mx.split(qkv, 3, -1)

    queries = queries.reshape(B, L, this.nHead, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, this.nHead, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, this.nHead, -1).transpose(0, 2, 1, 3)

    if (cache) {
      [keys, values] = cache.updateAndFetch(keys, values)
    }

    let output = mx.fast.scaledDotProductAttention(queries, keys, values, this.scale, mask)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, this.nEmbd)
    return this.cProj.forward(output)
  }
}

class MLP extends nn.Module {
  constructor(args) {
    super()

    this.nEmbd = args.nEmbd
    this.cFc = new nn.Linear(this.nEmbd, 4 * this.nEmbd)
    this.cProj = new nn.Linear(4 * this.nEmbd, this.nEmbd)
  }

  forward(x) {
    return this.cProj.forward(nn.geluApprox(this.cFc.forward(x)));
  }
}

class TransformerBlock extends nn.Module {
  constructor(args) {
    super()

    this.nHead = args.nHead
    this.nEmbd = args.nEmbd
    this.layerNormEpsilon = args.layerNormEpsilon
    this.attn = new Attention(args)
    this.mlp = new MLP(args)
    this.ln1 = new nn.LayerNorm(this.nEmbd, this.layerNormEpsilon)
    this.ln2 = new nn.LayerNorm(this.nEmbd, this.layerNormEpsilon)
  }

  forward(x, mask, cache) {
    let r = this.attn.forward(this.ln1.forward(x), mask, cache)
    let h = mx.add(x, r)
    r = this.mlp.forward(this.ln2.forward(h))
    return mx.add(h, r)
  }
}

class GPT2Model extends nn.Module {
  constructor(args) {
    super()

    this.nEmbd = args.nEmbd
    this.nPositions = args.nPositions
    this.vocabSize = args.vocabSize
    this.nLayer = args.nLayer
    this.layerNormEpsilon = args.layerNormEpsilon
    this.wte = new nn.Embedding(this.vocabSize, this.nEmbd)
    this.wpe = new nn.Embedding(this.nPositions, this.nEmbd)
    this.h = []
    for (let i = 0; i < args.nLayer; ++i)
      this.h.push(new TransformerBlock(args))
    this.lnF = new nn.LayerNorm(this.nEmbd, this.layerNormEpsilon)
  }

  forward(inputs, cache) {
    const L = inputs.shape[1]

    let hiddenStates = this.wte.forward(inputs)

    let mask
    if (hiddenStates.shape[1] > 1) {
      const positionIds = mx.arange(L)
      hiddenStates = mx.add(hiddenStates, this.wpe.forward(positionIds))
      mask = createAdditiveCausalMask(hiddenStates.shape[1], cache ? cache[0].offset : 0)
      mask = mask.astype(hiddenStates.dtype)
    }

    cache = cache ?? new Array(this.h.length)

    for (let i in this.h)
      hiddenStates = this.h[i].forward(hiddenStates, mask, cache[i])

    return this.lnF.forward(hiddenStates)
  }
}

export class Model extends nn.Module {
  constructor(obj) {
    const args = modelArgs(obj)
    super()

    this.args = args
    this.modelType = args.modelType
    this.model = new GPT2Model(args)
  }

  forward(inputs, cache) {
    const out = this.model.forward(inputs, cache)
    return this.model.wte.asLinear(out)
  }

  get layers() {
    return this.model.h
  }

  get headDim() {
    return Math.floor(this.args.nEmbd / this.args.nHead)
  }

  get nKVHeads() {
    return this.args.numKeyValueHeads
  }
}
