import {core as mx, nn} from '@frost-beta/mlx';
import {BaseModel, baseModelArgs, createAttentionMask} from '../base.js';
import {BaseKVCache} from '../kv-cache.js';

interface ModelArgs {
  modelType: 'qwen2';
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numHiddenLayers: number;
  numKeyValueHeads: number;
  rmsNormEps: number;
  ropeScaling?: {
    type: string;
    factor: number;
  };
  ropeTheta: number;
  ropeTraditional: boolean;
  tieWordEmbeddings: boolean;
  vocabSize: number;
};

function modelArgs(args: any): ModelArgs {
  args = Object.assign({
    ropeTheta: 1000000,
    ropeTraditional: false,
    tieWordEmbeddings: true,
  }, baseModelArgs(args));
  if (!args.numKeyValueHeads) {
    args.numKeyValueHeads = args.numAttentionHeads;
  }
  if (args.ropeScaling) {
    const requiredKeys = [ 'factor', 'type' ];
    if (!Object.keys(args.ropeScaling).every(key => requiredKeys.includes(key)))
      throw Error(`rope_scaling must contain keys ${requiredKeys}`);
    if (args.ropeScaling.type != 'linear')
      throw Error("rope_scaling 'type' currently only supports 'linear'");
  }
  return args;
}

class Attention extends nn.Module {
  nHeads: number;
  nKVHeads: number;
  scale: number;
  qProj: nn.Linear;
  kProj: nn.Linear;
  vProj: nn.Linear;
  oProj: nn.Linear;
  rope: nn.RoPE;

  constructor(args: ModelArgs) {
    super()

    const dim = args.hiddenSize;
    this.nHeads = args.numAttentionHeads;
    this.nKVHeads = args.numKeyValueHeads;

    const headDim = Math.floor(args.hiddenSize / this.nHeads);
    this.scale = headDim ** -0.5;

    this.qProj = new nn.Linear(dim, this.nHeads * headDim, true);
    this.kProj = new nn.Linear(dim, this.nKVHeads * headDim, true);
    this.vProj = new nn.Linear(dim, this.nKVHeads * headDim, true);
    this.oProj = new nn.Linear(this.nHeads * headDim, dim, false);

    const ropeScale = args.ropeScaling?.type == 'linear' ? 1 / args.ropeScaling.factor
                                                         : 1;
    this.rope = new nn.RoPE(headDim, args.ropeTraditional, args.ropeTheta, ropeScale);
  }

  forward(x: mx.array, mask: mx.array, cache?: BaseKVCache) {
    const [ B, L, D ] = x.shape;

    let queries = this.qProj.forward(x);
    let keys = this.kProj.forward(x);
    let values = this.vProj.forward(x);

    // Prepare the queries, keys and values for the attention computation.
    queries = queries.reshape(B, L, this.nHeads, -1).transpose(0, 2, 1, 3);
    keys = keys.reshape(B, L, this.nKVHeads, -1).transpose(0, 2, 1, 3);
    values = values.reshape(B, L, this.nKVHeads, -1).transpose(0, 2, 1, 3);

    if (cache) {
      queries = this.rope.forward(queries, cache.offset);
      keys = this.rope.forward(keys, cache.offset);
      [ keys, values ] = cache.updateAndFetch(keys, values);
    } else {
      queries = this.rope.forward(queries);
      keys = this.rope.forward(keys);
    }

    let output = mx.fast.scaledDotProductAttention(queries, keys, values, this.scale, mask);
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1);
    return this.oProj.forward(output);
  }
}

class MLP extends nn.Module {
  gateProj: nn.Linear;
  downProj: nn.Linear;
  upProj: nn.Linear;

  constructor(dim: number, hiddenDim: number) {
    super();
    this.gateProj = new nn.Linear(dim, hiddenDim, false);
    this.downProj = new nn.Linear(hiddenDim, dim, false);
    this.upProj = new nn.Linear(dim, hiddenDim, false);
  }

  forward(x: mx.array) {
    return this.downProj.forward(mx.multiply(nn.silu(this.gateProj.forward(x)),
                                             this.upProj.forward(x)));
  }
}

class TransformerBlock extends nn.Module {
  selfAttn: Attention;
  mlp: MLP;
  inputLayernorm: nn.RMSNorm;
  postAttentionLayernorm: nn.RMSNorm;

  constructor(args: ModelArgs) {
    super()
    this.selfAttn = new Attention(args);
    this.mlp = new MLP(args.hiddenSize, args.intermediateSize);
    this.inputLayernorm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps);
    this.postAttentionLayernorm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps);
  }

  forward(x: mx.array, mask: mx.array, cache?: BaseKVCache) {
    const r = this.selfAttn.forward(this.inputLayernorm.forward(x), mask, cache);
    const h = mx.add(x, r);
    const r2 = this.mlp.forward(this.postAttentionLayernorm.forward(h));
    return mx.add(h, r2);
  }
}

class Qwen2Model extends nn.Module {
  embedTokens: nn.Embedding;
  layers: TransformerBlock[];
  norm: nn.RMSNorm;

  constructor(args: ModelArgs) {
    super();
    this.embedTokens = new nn.Embedding(args.vocabSize, args.hiddenSize);
    this.layers = [];
    for (let i = 0; i < args.numHiddenLayers; ++i)
      this.layers.push(new TransformerBlock(args));
    this.norm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps);
  }

  forward(embeddings: mx.array, cache?: BaseKVCache[]) {
    let h = embeddings;
    const mask = createAttentionMask(h, cache);
    for (let i in this.layers)
      h = this.layers[i].forward(h, mask, cache ? cache[i] : undefined);
    return this.norm.forward(h);
  }
}

export class Model extends BaseModel {
  args: ModelArgs;
  model: Qwen2Model;
  lmHead: nn.Linear;

  constructor(json: any) {
    const args = modelArgs(json);
    super();

    this.args = args;
    this.model = new Qwen2Model(args);
    if (!args.tieWordEmbeddings)
      this.lmHead = new nn.Linear(args.hiddenSize, args.vocabSize, false);
  }

  override computeTextEmbeddings(inputs: mx.array): mx.array {
    return this.model.embedTokens.forward(inputs);
  }

  override forwardEmbeddings(embeddings: mx.array, cache?: BaseKVCache[]): mx.array {
    const out = this.model.forward(embeddings, cache);
    if (this.args.tieWordEmbeddings)
      return this.model.embedTokens.asLinear(out);
    else
      return this.lmHead.forward(out);
  }

  override get layers() {
    return this.model.layers;
  }

  override get headDim() {
    return Math.floor(this.args.hiddenSize / this.args.numAttentionHeads);
  }

  override get nKVHeads() {
    return this.args.numKeyValueHeads;
  }
}
