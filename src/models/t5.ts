import {core as mx, nn} from '@frost-beta/mlx';
import {BaseModel, baseModelArgs, createAttentionMask} from '../base.js';
import {BaseKVCache} from '../kv-cache.js';

export interface ModelArgs {
  classifierDropout: number;
  dFf: number;
  dKv: number;
  dModel: number;
  denseActFn: string;
  dropoutRate: number;
  eosTokenId: number;
  feedForwardProj: string;
  initializerFactor: number;
  isEncoderDecoder: boolean;
  isGatedAct: boolean;
  layerNormEpsilon: number;
  numDecoderLayers?: number;
  numHeads: number;
  numLayers: number;
  padTokenId: number;
  relativeAttentionMaxDistance: number;
  relativeAttentionNumBuckets: number;
  useCache: boolean;
  vocabSize: number;
}

export function modelArgs(json: any): ModelArgs {
  const args = Object.assign({
    classifierDropout: 0.0,
    dFf: 2048,
    dKv: 64,
    dModel: 512,
    dropoutRate: 0.1,
    eosTokenId: 1,
    feedForwardProj: 'relu',
    initializerFactor: 1.0,
    isEncoderDecoder: true,
    layerNormEpsilon: 1e-6,
    numHeads: 8,
    numLayers: 6,
    padTokenId: 0,
    relativeAttentionMaxDistance: 128,
    relativeAttentionNumBuckets: 32,
    useCache: true,
    vocabSize: 32128,
  }, baseModelArgs(json));
  args.denseActFn = args.feedForwardProj.split('-').at(-1);
  args.isGatedAct = args.feedForwardProj.startsWith('gated-');
  return args;
}

class RelativeAttentionBias extends nn.Module {
  embeddings: nn.Embedding;

  constructor(public args: ModelArgs, public bidirectional: boolean) {
    super();
    this.embeddings = new nn.Embedding(args.relativeAttentionNumBuckets, args.numHeads);
  }

  forward(queryLength: number, keyLength: number, offset = 0) {
    const contextPosition = mx.arange(offset, queryLength).index(mx.Slice(), mx.newaxis);
    const memoryPosition = mx.arange(keyLength).index(mx.newaxis, mx.Slice());

    const relativePosition = mx.subtract(memoryPosition, contextPosition);
    const relativePositionBucket = this.relativePositionBucket(
      relativePosition,
      this.bidirectional,
      this.args.relativeAttentionNumBuckets,
      this.args.relativeAttentionMaxDistance);
    const values = this.embeddings.forward(relativePositionBucket);
    return values.transpose(2, 0, 1);
  }

  relativePositionBucket(relativePosition: mx.array, bidirectional: boolean, numBuckets: number, maxDistance: number) {
    let relativeBuckets = mx.array(0, mx.int16);
    if (bidirectional) {
      numBuckets /= 2;
      relativeBuckets = mx.add(relativeBuckets,
                               mx.multiply(mx.greater(relativePosition, 0).astype(mx.int16),
                                           mx.array(numBuckets, mx.int16)));
      relativePosition = mx.abs(relativePosition);
    } else {
      relativePosition = mx.negative(mx.minimum(relativePosition,
                                                mx.zerosLike(relativePosition)));
    }

    const maxExact = numBuckets / 2;
    const isSmall = mx.less(relativePosition, maxExact);

    const scale = (numBuckets - maxExact) / Math.log(maxDistance / maxExact);
    let relativePositionIfLarge = mx.add(mx.array(maxExact, mx.int16),
                                         mx.multiply(mx.log(mx.divide(relativePosition.astype(mx.float32),
                                                                      maxExact)),
                                                     scale).astype(mx.int16));
    relativePositionIfLarge = mx.minimum(relativePositionIfLarge, numBuckets - 1);
    relativeBuckets = mx.add(relativeBuckets,
                             mx.where(isSmall, relativePosition, relativePositionIfLarge));
    return relativeBuckets;
  }
}

const ACT2FN = {
  relu: nn.relu,
  gelu: nn.gelu,
  silu: nn.silu,
};

class DenseActDense extends nn.Module {
  wi: nn.Linear;
  wo: nn.Linear;
  dropout: nn.Dropout;
  act: (x: mx.array) => mx.array;

  constructor(args: ModelArgs) {
    super();
    this.wi = new nn.Linear(args.dModel, args.dFf, false);
    this.wo = new nn.Linear(args.dFf, args.dModel, false);
    this.dropout = new nn.Dropout(args.dropoutRate);
    this.act = ACT2FN[args.denseActFn as keyof typeof ACT2FN];
  }

  forward(x: mx.array) {
    x = this.wi.forward(x);
    x = this.act(x);
    x = this.dropout.forward(x);
    x = this.wo.forward(x);
    return x;
  }
}

class DenseGatedActDense extends nn.Module {
  wi_0: nn.Linear;
  wi_1: nn.Linear;
  wo: nn.Linear;
  dropout: nn.Dropout;
  act: (x: mx.array) => mx.array;

  constructor(args: ModelArgs) {
    super();
    this.wi_0 = new nn.Linear(args.dModel, args.dFf, false);
    this.wi_1 = new nn.Linear(args.dModel, args.dFf, false);
    this.wo = new nn.Linear(args.dFf, args.dModel, false);
    this.dropout = new nn.Dropout(args.dropoutRate);
    this.act = ACT2FN[args.denseActFn as keyof typeof ACT2FN];
  }

  forward(x: mx.array) {
    x = mx.multiply(this.act(this.wi_0.forward(x)),
                    this.wi_1.forward(x));
    x = this.dropout.forward(x);
    x = this.wo.forward(x);
    return x;
  }
}

class LayerFF extends nn.Module {
  DenseReluDense: DenseActDense | DenseGatedActDense;
  layerNorm: nn.RMSNorm;
  dropout: nn.Dropout;

  constructor(args: ModelArgs) {
    super();
    this.DenseReluDense = args.isGatedAct ? new DenseGatedActDense(args)
                                          : new DenseActDense(args);
    this.layerNorm = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.dropout = new nn.Dropout(args.dropoutRate);
  }

  forward(x: mx.array) {
    let f = this.layerNorm.forward(x);
    f = this.DenseReluDense.forward(f);
    x = mx.add(x, this.dropout.forward(f));
    return x;
  }
}

class Attention extends nn.Module {
  q: nn.Linear;
  k: nn.Linear;
  v: nn.Linear;
  o: nn.Linear;

  constructor(public args: ModelArgs,
              public isDecoder: boolean,
              public hasRelativeAttentionBias = false) {
    super();
    const innderDim = args.numHeads * args.dKv;
    this.q = new nn.Linear(args.dModel, innderDim, false);
    this.k = new nn.Linear(args.dModel, innderDim, false);
    this.v = new nn.Linear(args.dModel, innderDim, false);
    this.o = new nn.Linear(innderDim, args.dModel, false);
  }

  forward(queries: mx.array, keys: mx.array, values: mx.array, mask?: mx.array, cache?: BaseKVCache) {
    queries = this.q.forward(queries);
    keys = this.k.forward(keys);
    values = this.v.forward(values);

    const {numHeads} = this.args;
    const [ B, L, D ] = queries.shape;
    const [  , S,   ] = keys.shape;

    queries = queries.reshape(B, L, numHeads, -1).transpose(0, 2, 1, 3);
    keys = keys.reshape(B, S, numHeads, -1).transpose(0, 2, 3, 1);
    values = values.reshape(B, S, numHeads, -1).transpose(0, 2, 1, 3);

    if (cache)
      [ keys, values ] = cache.updateAndFetch(keys, values);

    const scale = Math.sqrt(1 / queries.shape.at(-1));
    let output = mx.fast.scaledDotProductAttention(queries.astype(mx.float32), keys, values, scale, mask).astype(values.dtype);
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1);
    return this.o.forward(output);
  }
}

class LayerSelfAttention extends nn.Module {
  SelfAttention: Attention;
  layerNorm: nn.RMSNorm;
  dropout: nn.Dropout;

  constructor(args: ModelArgs, isDecoder: boolean, hasRelativeAttentionBias = false) {
    super();
    this.SelfAttention = new Attention(args, isDecoder, hasRelativeAttentionBias);
    this.layerNorm = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.dropout = new nn.Dropout(args.dropoutRate);
  }

  forward(x: mx.array, keys?: mx.array, values?: mx.array, mask?: mx.array, cache?: BaseKVCache) {
    let y = this.layerNorm.forward(x);
    y = this.SelfAttention.forward(y, keys ?? y, values ?? y, mask, cache);
    return mx.add(x, this.dropout.forward(y));
  }
}

class LayerCrossAttention extends nn.Module {
  EncDecAttention: Attention;
  layerNorm: nn.RMSNorm;
  dropout: nn.Dropout;

  constructor(args: ModelArgs) {
    super();
    this.EncDecAttention = new Attention(args, true, false);
    this.layerNorm = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.dropout = new nn.Dropout(args.dropoutRate);
  }

  forward(x: mx.array, keys: mx.array, values: mx.array, mask?: mx.array, cache?: BaseKVCache) {
    let y = this.layerNorm.forward(x);
    y = this.EncDecAttention.forward(x, keys, values, mask, cache);
    return mx.add(x, this.dropout.forward(y));
  }
}

class EncoderBlock extends nn.Module {
  attention: Attention;
  ln1: nn.RMSNorm;
  ln2: nn.RMSNorm;
  dense: DenseActDense | DenseGatedActDense;

  constructor(args: ModelArgs) {
    super();
    this.attention = new Attention(args, false);
    this.ln1 = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.ln2 = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.dense = args.isGatedAct ? new DenseGatedActDense(args)
                                 : new DenseActDense(args);
  }

  forward(x: mx.array, mask?: mx.array) {
    let y = this.ln1.forward(x);
    y = this.attention.forward(y, y, y, mask);
    x = mx.add(x, y);
    y = this.ln2.forward(x);
    y = this.dense.forward(y);
    x = mx.add(x, y);
    return x;
  }
}

class Encoder extends nn.Module {
  layers: EncoderBlock[] = [];
  ln: nn.RMSNorm;
  relativeAttentionBias: RelativeAttentionBias;
  dropout: nn.Dropout;

  constructor(args: ModelArgs) {
    super();
    for (let i = 0; i < args.numLayers; ++i)
      this.layers.push(new EncoderBlock(args));
    this.ln = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.relativeAttentionBias = new RelativeAttentionBias(args, true);
    this.dropout = new nn.Dropout(args.dropoutRate);
  }

  forward(x: mx.array) {
    const L = x.shape[1];
    const positionBias = this.relativeAttentionBias.forward(L, L);
    for (const layer of this.layers)
      x = layer.forward(x, positionBias);
    x = this.ln.forward(x);
    x = this.dropout.forward(x);
    return x;
  }
}

class DecoderBlock extends nn.Module {
  selfAttention: Attention;
  crossAttention: Attention;
  ln1: nn.RMSNorm;
  ln2: nn.RMSNorm;
  ln3: nn.RMSNorm;
  dense: DenseActDense | DenseGatedActDense;

  constructor(args: ModelArgs) {
    super();
    this.selfAttention = new Attention(args, true);
    this.crossAttention = new Attention(args, true);
    this.ln1 = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.ln2 = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.ln3 = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.dense = args.isGatedAct ? new DenseGatedActDense(args)
                                 : new DenseActDense(args);
  }

  forward(x: mx.array, memory: mx.array, mask?: mx.array, memoryMask?: mx.array, cache?: BaseKVCache) {
    let y = this.ln1.forward(x);
    y = this.selfAttention.forward(y, y, y, mask, cache);
    x = mx.add(x, y);
    y = this.ln2.forward(x);
    y = this.crossAttention.forward(y, memory, memory, memoryMask);
    x = mx.add(x, y);
    y = this.ln3.forward(x);
    y = this.dense.forward(x);
    x = mx.add(x, y);
    return x;
  }
}

class Decoder extends nn.Module {
  layers: DecoderBlock[] = [];
  ln: nn.RMSNorm;
  relativeAttentionBias: RelativeAttentionBias;
  dropout: nn.Dropout;

  constructor(args: ModelArgs) {
    super();
    for (let i = 0; i < args.numDecoderLayers; ++i)
      this.layers.push(new DecoderBlock(args));
    this.ln = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.relativeAttentionBias = new RelativeAttentionBias(args, false);
    this.dropout = new nn.Dropout(args.dropoutRate);
  }

  forward(x: mx.array, memory: mx.array, mask?: mx.array, memoryMask?: mx.array, cache?: BaseKVCache[]) {
    const offset = cache ? cache[0].offset : 0;
    const T = offset + x.shape[1];
    const positionBias = this.relativeAttentionBias.forward(T, T, offset);
    if (mask)
      mask = mx.add(mask, positionBias);
    else
      mask = positionBias;
    for (let i in this.layers)
      x = this.layers[i].forward(x, memory, memoryMask, mask, cache ? cache[i] : undefined);
    x = this.ln.forward(x);
    x = this.dropout.forward(x);
    return x;
  }
}

export class Model extends BaseModel {
  args: ModelArgs;
  shared: nn.Embedding;
  encoder: Encoder;
  decoder: Decoder;
  lmHead: nn.Linear;

  constructor(json: any) {
    const args = modelArgs(json);
    super();

    this.args = args;
    this.shared = new nn.Embedding(args.vocabSize, args.dModel);
    this.encoder = new Encoder(args);
    this.decoder = new Decoder(args);
    this.lmHead = new nn.Linear(args.dModel, args.vocabSize, false);
  }

  override computeTextEmbeddings(inputs: mx.array): mx.array {
    return this.shared.forward(inputs);
  }

  override forwardEmbeddings(embeddings: mx.array, cache?: BaseKVCache[]): mx.array {
    return this.decode(mx.array([ 0 ], mx.int16), this.encoder.forward(embeddings), cache);
  }

  override get layers() {
    return this.decoder.layers;
  }

  override get headDim() {
    return this.args.dKv;
  }

  override get nKVHeads() {
    return this.args.numHeads;
  }

  decode(inputs: mx.array, memory: mx.array, cache?: BaseKVCache[]) {
    const mask = createAttentionMask(inputs, cache);
    const y = this.decoder.forward(inputs, memory, mask, undefined, cache);
    return this.lmHead.forward(y);
  }
}
