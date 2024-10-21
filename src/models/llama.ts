import {core as mx, nn} from '@frost-beta/mlx';
import {BaseModel, baseModelArgs, createAttentionMask} from '../base.js';
import {KVCacheOptions, BaseKVCache} from '../kv-cache.js';

export interface RopeScaling {
  type?: string;
  ropeType?: string;
  factor?: number;
  lowFreqFactor?: number;
  highFreqFactor?: number;
  originalMaxPositionEmbeddings?: number;
}

export interface ModelArgs {
  modelType: 'gemma' | 'llama' | 'qwen2';
  attentionBias: boolean;
  attentionOutProjectionBias: boolean;
  headDim?: number;
  hiddenAct: 'gelu' | 'geluApprox' | 'silu';
  hiddenSize: number;
  intermediateSize: number;
  maxPositionEmbeddings: number;
  mlpBias: boolean;
  numAttentionHeads: number;
  numHiddenLayers: number;
  numKeyValueHeads: number;
  rmsNormEps: number;
  ropeScaling?: RopeScaling;
  ropeTheta: number;
  ropeTraditional: boolean;
  tieWordEmbeddings: boolean;
  vocabSize: number;
};

export function modelArgs(json: any): ModelArgs {
  const args = Object.assign({
    attentionBias: false,
    mlpBias: false,
    ropeTheta: 10000,
    ropeTraditional: false,
    tieWordEmbeddings: true,
  }, baseModelArgs(json));
  if (args.attentionOutProjectionBias === undefined)
    args.attentionOutProjectionBias = args.attentionBias;
  if (args.hiddenAct == undefined)
    args.hiddenAct = 'silu';
  if (args.numKeyValueHeads === undefined)
    args.numKeyValueHeads = args.numAttentionHeads;
  if (args.ropeScaling) {
    if (!args.ropeScaling.factor)
      throw new Error('rope_scaling must contain "factor"')
    const ropeType = args.ropeScaling.type || args.ropeScaling.ropeType;
    if (!ropeType)
      throw new Error('rope_scaling must contain either "type" or "rope_type"');
    if (!['linear', 'dynamic', 'llama3'].includes(ropeType))
      throw new Error('rope_scaling "type" currently only supports "linear", "dynamic" or "llama3"');
  }
  return args;
}

export class GemmaRMSNorm extends nn.RMSNorm {
  constructor(dims: number, eps: number) {
    super(dims, eps);
  }

  override forward(x: mx.array): mx.array {
    return mx.fast.rmsNorm(x, mx.add(1, this.weight), this.eps);
  }
}

class DynamicNTKScalingRoPE extends nn.Module {
  #freqs?: mx.array;

  constructor(public dims: number,
              public maxPositionEmbeddings = 2048,
              public traditional = false,
              public base = 10000,
              public scale = 1.0,
              public ropeType = 'default',
              public ropeScaling?: RopeScaling) {
    super();
    this.computeBaseFreq();
  }

  private computeBaseFreq() {
    if (this.ropeType != 'llama3')
      return;

    const factor = this.ropeScaling?.factor ?? 1;
    const lowFreqFactor = this.ropeScaling?.lowFreqFactor ?? 1;
    const highFreqFactor = this.ropeScaling?.highFreqFactor ?? 4;
    const oldContextLen = this.ropeScaling?.originalMaxPositionEmbeddings ?? 8192;

    const lowFreqWavelen = oldContextLen / lowFreqFactor;
    const highFreqWavelen = oldContextLen / highFreqFactor;

    let freqs = mx.power(this.base, mx.divide(mx.arange(0, this.dims, 2), this.dims));
    const wavelens = mx.multiply(2 * Math.PI, freqs);

    freqs = mx.where(mx.greater(wavelens, lowFreqWavelen), mx.multiply(freqs, factor), freqs);
    const isMediumFreq = mx.bitwiseAnd(mx.greater(wavelens, highFreqWavelen),
                                       mx.less(wavelens, lowFreqWavelen));
    const smoothFactors = mx.divide(mx.subtract(mx.divide(oldContextLen, wavelens),
                                                lowFreqFactor),
                                    highFreqFactor - lowFreqFactor);
    const smoothFreqs = mx.divide(freqs,
                                  mx.add(mx.divide(mx.subtract(1, smoothFactors),
                                                   factor),
                                         smoothFactors));
    this.#freqs = mx.where(isMediumFreq, smoothFreqs, freqs);
    this.base = undefined;
  }

  forward(x: mx.array, offset = 0) {
    return mx.fast.rope(x, this.dims, this.traditional, this.base, this.scale, offset, this.#freqs);
  }
}

function initializeRoPE(args: ModelArgs) {
  const headDim = args.headDim ?? args.hiddenSize / args.numAttentionHeads;

  const ropeScaling = args.ropeScaling;
  let ropeType = 'default';
  let ropeScale = 1.0;

  if (ropeScaling) {
    ropeType = (ropeScaling.type ?? ropeScaling.ropeType) ?? 'default';
    if (ropeType == 'linear')
      ropeScale = 1 / ropeScaling.factor;
    else if (ropeType == 'llama3')
      ropeScale = 1;
  }

  if (ropeType == 'llama3') {
    return new DynamicNTKScalingRoPE(headDim,
                                     args.maxPositionEmbeddings,
                                     args.ropeTraditional,
                                     args.ropeTheta,
                                     ropeScale,
                                     ropeType,
                                     ropeScaling);
  } else {
    return new nn.RoPE(headDim, args.ropeTraditional, args.ropeTheta, ropeScale);
  }
}

class Attention extends nn.Module {
  nHeads: number;
  nKVHeads: number;
  scale: number;
  qProj: nn.Linear;
  kProj: nn.Linear;
  vProj: nn.Linear;
  oProj: nn.Linear;
  rope: nn.RoPE | DynamicNTKScalingRoPE;

  constructor(args: ModelArgs) {
    super();
    const dim = args.hiddenSize;
    this.nHeads = args.numAttentionHeads;
    this.nKVHeads = args.numKeyValueHeads;

    const headDim = Math.floor(args.hiddenSize / this.nHeads);
    this.scale = headDim ** -0.5;

    this.qProj = new nn.Linear(dim, this.nHeads * headDim, args.attentionBias);
    this.kProj = new nn.Linear(dim, this.nKVHeads * headDim, args.attentionBias);
    this.vProj = new nn.Linear(dim, this.nKVHeads * headDim, args.attentionBias);
    this.oProj = new nn.Linear(this.nHeads * headDim, dim, args.attentionOutProjectionBias);

    this.rope = initializeRoPE(args);
  }

  forward(x: mx.array, mask?: mx.array, cache?: BaseKVCache) {
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
  activation: (x: mx.array) => mx.array;

  constructor(args: ModelArgs) {
    super();

    const dim = args.hiddenSize;
    const hiddenDim = args.intermediateSize;
    const mlpBias = args.mlpBias;

    this.gateProj = new nn.Linear(dim, hiddenDim, mlpBias);
    this.downProj = new nn.Linear(hiddenDim, dim, mlpBias);
    this.upProj = new nn.Linear(dim, hiddenDim, mlpBias);
    this.activation = nn[args.hiddenAct as 'silu'];
  }

  forward(x: mx.array) {
    return this.downProj.forward(mx.multiply(this.activation(this.gateProj.forward(x)),
                                             this.upProj.forward(x)));
  }
}

class TransformerBlock extends nn.Module {
  selfAttn: Attention;
  mlp: MLP;
  inputLayernorm: nn.RMSNorm;
  postAttentionLayernorm: nn.RMSNorm;

  constructor(args: ModelArgs) {
    super();
    this.selfAttn = new Attention(args);
    this.mlp = new MLP(args);
    const norm = args.modelType == 'gemma' ? GemmaRMSNorm : nn.RMSNorm;
    this.inputLayernorm = new norm(args.hiddenSize, args.rmsNormEps);
    this.postAttentionLayernorm = new norm(args.hiddenSize, args.rmsNormEps);
  }

  forward(x: mx.array, mask: mx.array, cache?: BaseKVCache) {
    const r = this.selfAttn.forward(this.inputLayernorm.forward(x), mask, cache);
    const h = mx.add(x, r);
    const r2 = this.mlp.forward(this.postAttentionLayernorm.forward(h));
    return mx.add(h, r2);
  }
}

class LlamaModel extends nn.Module {
  embedTokens: nn.Embedding;
  layers: TransformerBlock[];
  norm: nn.RMSNorm;
  gemmaNormalizer?: number;

  constructor(args: ModelArgs) {
    super();
    this.embedTokens = new nn.Embedding(args.vocabSize, args.hiddenSize);
    this.layers = [];
    for (let i = 0; i < args.numHiddenLayers; ++i)
      this.layers.push(new TransformerBlock(args));
    if (args.modelType == 'gemma') {
      this.norm = new GemmaRMSNorm(args.hiddenSize, args.rmsNormEps);
      this.gemmaNormalizer = args.hiddenSize ** 0.5;
    } else {
      this.norm = new nn.RMSNorm(args.hiddenSize, args.rmsNormEps);
    }
  }

  forward(embeddings: mx.array, cache?: BaseKVCache[]) {
    let h = embeddings;
    if (this.gemmaNormalizer !== undefined)
      h = mx.multiply(h, this.gemmaNormalizer);
    const mask = createAttentionMask(h, cache);
    for (let i in this.layers)
      h = this.layers[i].forward(h, mask, cache ? cache[i] : undefined);
    return this.norm.forward(h);
  }
}

export class Model extends BaseModel {
  args: ModelArgs;
  model: LlamaModel;
  lmHead: nn.Linear;

  constructor(json: any) {
    super();
    const args = modelArgs(json);
    this.args = args;
    this.model = new LlamaModel(args);
    if (!args.tieWordEmbeddings)
      this.lmHead = new nn.Linear(args.hiddenSize, args.vocabSize, false);
  }

  override computeTextEmbeddings(inputs: mx.array): mx.array {
    return this.model.embedTokens.forward(inputs);
  }

  override decodeEmbeddings(embeddings: mx.array, memory?: mx.array, cache?: BaseKVCache[]): mx.array {
    if (memory)
      throw new Error('This model has no encoder.');
    const out = this.model.forward(embeddings, cache);
    if (this.args.tieWordEmbeddings)
      return this.model.embedTokens.asLinear(out);
    else
      return this.lmHead.forward(out);
  }

  override getDecoderKVCacheOptions(): KVCacheOptions {
    return {nLayers: this.model.layers.length};
  }
}
