import {core as mx, nn} from '@frost-beta/mlx';
import {BaseModel, baseModelArgs, createAttentionMask} from '../base.js';
import {KVCacheOptions, BaseKVCache} from '../kv-cache.js';

export interface ModelArgs {
  classifierDropout: number;
  dFf: number;
  dKv: number;
  dModel: number;
  decoderStartTokenId: number;
  denseActFn: string;
  dropoutRate: number;
  eosTokenId: number;
  feedForwardProj: string;
  initializerFactor: number;
  isGatedAct: boolean;
  layerNormEpsilon: number;
  numDecoderLayers: number;
  numHeads: number;
  numLayers: number;
  padTokenId: number;
  relativeAttentionMaxDistance: number;
  relativeAttentionNumBuckets: number;
  tieWordEmbeddings: boolean;
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
    layerNormEpsilon: 1e-6,
    numHeads: 8,
    numLayers: 6,
    padTokenId: 0,
    relativeAttentionMaxDistance: 128,
    relativeAttentionNumBuckets: 32,
    tieWordEmbeddings: true,
    vocabSize: 32128,
  }, baseModelArgs(json));
  if (args.decoderStartTokenId === undefined) {
    args.decoderStartTokenId = args.padTokenId;
    if (args.decoderStartTokenId === undefined)
      throw new Error('Must provide "decoder_start_token_id" or "pad_token_id"');
  }
  if (!args.numDecoderLayers) {
    args.numDecoderLayers = args.numLayers;
  }
  args.denseActFn = args.feedForwardProj.split('-').at(-1);
  args.isGatedAct = args.feedForwardProj.startsWith('gated-');
  return args;
}

class RelativeAttentionBias extends nn.Module {
  constructor(public args: ModelArgs, public bidirectional: boolean) {
    super();
  }

  forward(embeddings: nn.Embedding, queryLength: number, keyLength: number, offset = 0) {
    const contextPosition = mx.arange(offset, queryLength, 1, mx.int16).index(mx.Slice(), mx.newaxis);
    const memoryPosition = mx.arange(keyLength, mx.int16).index(mx.newaxis, mx.Slice());

    const relativePosition = mx.subtract(memoryPosition, contextPosition);
    const relativePositionBucket = this.relativePositionBucket(
      relativePosition,
      this.bidirectional,
      this.args.relativeAttentionNumBuckets,
      this.args.relativeAttentionMaxDistance);
    const values = embeddings.forward(relativePositionBucket);
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
    relativePositionIfLarge = mx.minimum(relativePositionIfLarge,
                                         mx.array(numBuckets - 1, mx.int16));
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

  constructor(args: ModelArgs) {
    super();
    this.DenseReluDense = args.isGatedAct ? new DenseGatedActDense(args)
                                          : new DenseActDense(args);
    this.layerNorm = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
  }

  forward(x: mx.array) {
    let y = this.layerNorm.forward(x);
    y = this.DenseReluDense.forward(y);
    return y;
  }
}

class Attention extends nn.Module {
  q: nn.Linear;
  k: nn.Linear;
  v: nn.Linear;
  o: nn.Linear;
  relativeAttentionBias?: nn.Embedding;

  constructor(public args: ModelArgs, public hasRelativeAttentionBias = false) {
    super();
    const innderDim = args.numHeads * args.dKv;
    this.q = new nn.Linear(args.dModel, innderDim, false);
    this.k = new nn.Linear(args.dModel, innderDim, false);
    this.v = new nn.Linear(args.dModel, innderDim, false);
    this.o = new nn.Linear(innderDim, args.dModel, false);
    if (hasRelativeAttentionBias)
      this.relativeAttentionBias = new nn.Embedding(args.relativeAttentionNumBuckets, args.numHeads);
  }

  forward(queries: mx.array, keys: mx.array, values: mx.array, mask?: mx.array, cache?: BaseKVCache) {
    queries = this.q.forward(queries);
    keys = this.k.forward(keys);
    values = this.v.forward(values);

    const {numHeads} = this.args;
    const [ B, L, D ] = queries.shape;
    const [  , S,   ] = keys.shape;

    queries = queries.reshape(B, L, numHeads, -1).transpose(0, 2, 1, 3);
    keys = keys.reshape(B, S, numHeads, -1).transpose(0, 2, 1, 3);
    values = values.reshape(B, S, numHeads, -1).transpose(0, 2, 1, 3);

    if (cache)
      [ keys, values ] = cache.updateAndFetch(keys, values);

    let scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2));
    if (mask)
      scores = mx.add(scores, mask.astype(scores.dtype));
    scores = mx.softmax(scores.astype(mx.float32), -1).astype(scores.dtype);
    const output = mx.matmul(scores, values).transpose(0, 2, 1, 3).reshape(B, L, -1);
    return this.o.forward(output);
  }
}

class LayerSelfAttention extends nn.Module {
  SelfAttention: Attention;
  layerNorm: nn.RMSNorm;

  constructor(args: ModelArgs, hasRelativeAttentionBias = false) {
    super();
    this.SelfAttention = new Attention(args, hasRelativeAttentionBias);
    this.layerNorm = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
  }

  forward(x: mx.array, mask?: mx.array, cache?: BaseKVCache) {
    let y = this.layerNorm.forward(x);
    y = this.SelfAttention.forward(y, y, y, mask, cache);
    return y;
  }
}

class LayerCrossAttention extends nn.Module {
  EncDecAttention: Attention;
  layerNorm: nn.RMSNorm;

  constructor(args: ModelArgs, hasRelativeAttentionBias = false) {
    super();
    this.EncDecAttention = new Attention(args, hasRelativeAttentionBias);
    this.layerNorm = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
  }

  forward(x: mx.array, memory: mx.array, mask?: mx.array, cache?: BaseKVCache) {
    let y = this.layerNorm.forward(x);
    y = this.EncDecAttention.forward(y, memory, memory, mask, cache);
    return y;
  }
}

class EncoderBlock extends nn.Module {
  layer: [ LayerSelfAttention, LayerFF ];

  constructor(args: ModelArgs, hasRelativeAttentionBias = false) {
    super();
    this.layer = [
      new LayerSelfAttention(args, hasRelativeAttentionBias),
      new LayerFF(args),
    ];
  }

  forward(x: mx.array, mask?: mx.array) {
    let y = this.layer[0].forward(x, mask);
    x = mx.add(x, y);
    y = this.layer[1].forward(x);
    x = mx.add(x, y);
    return x;
  }
}

class Encoder extends nn.Module {
  block: EncoderBlock[] = [];
  finalLayerNorm: nn.RMSNorm;
  relativeAttentionBias: RelativeAttentionBias;
  dropout: nn.Dropout;

  constructor(args: ModelArgs) {
    super();
    for (let i = 0; i < args.numLayers; ++i)
      this.block.push(new EncoderBlock(args, i == 0));
    this.finalLayerNorm = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.relativeAttentionBias = new RelativeAttentionBias(args, true);
    this.dropout = new nn.Dropout(args.dropoutRate);
  }

  forward(x: mx.array) {
    const L = x.shape[1];
    const embeddings = this.block[0].layer[0].SelfAttention.relativeAttentionBias;
    const positionBias = this.relativeAttentionBias.forward(embeddings, L, L);
    for (const layer of this.block)
      x = layer.forward(x, positionBias);
    x = this.finalLayerNorm.forward(x);
    x = this.dropout.forward(x);
    return x;
  }
}

class DecoderBlock extends nn.Module {
  layer: [ LayerSelfAttention, LayerCrossAttention, LayerFF ];

  constructor(args: ModelArgs, hasRelativeAttentionBias = false) {
    super();
    this.layer = [
      new LayerSelfAttention(args, hasRelativeAttentionBias),
      new LayerCrossAttention(args, hasRelativeAttentionBias),
      new LayerFF(args),
    ];
  }

  forward(x: mx.array, memory: mx.array, mask?: mx.array, memoryMask?: mx.array, cache?: BaseKVCache) {
    let y = this.layer[0].forward(x, mask, cache);
    x = mx.add(x, y);
    y = this.layer[1].forward(x, memory, memoryMask);
    x = mx.add(x, y);
    y = this.layer[2].forward(x);
    x = mx.add(x, y);
    return x;
  }
}

class Decoder extends nn.Module {
  block: DecoderBlock[] = [];
  finalLayerNorm: nn.RMSNorm;
  relativeAttentionBias: RelativeAttentionBias;
  dropout: nn.Dropout;

  constructor(args: ModelArgs) {
    super();
    for (let i = 0; i < args.numDecoderLayers; ++i)
      this.block.push(new DecoderBlock(args, i == 0));
    this.finalLayerNorm = new nn.RMSNorm(args.dModel, args.layerNormEpsilon);
    this.relativeAttentionBias = new RelativeAttentionBias(args, false);
    this.dropout = new nn.Dropout(args.dropoutRate);
  }

  forward(x: mx.array, memory: mx.array, mask?: mx.array, memoryMask?: mx.array, cache?: BaseKVCache[]) {
    const offset = cache ? cache[0].offset : 0;
    const T = offset + x.shape[1];
    const embeddings = this.block[0].layer[0].SelfAttention.relativeAttentionBias;
    const positionBias = this.relativeAttentionBias.forward(embeddings, T, T, offset);
    if (mask)
      mask = mx.add(mask, positionBias);
    else
      mask = positionBias;
    for (let i in this.block)
      x = this.block[i].forward(x, memory, mask, memoryMask, cache ? cache[i] : undefined);
    x = this.finalLayerNorm.forward(x);
    x = this.dropout.forward(x);
    return x;
  }
}

export class Model extends BaseModel {
  args: ModelArgs;
  shared: nn.Embedding;
  encoder: Encoder;
  decoder: Decoder;
  lmHead?: nn.Linear;

  constructor(json: any) {
    super();
    const args = modelArgs(json);
    this.args = args;
    this.shared = new nn.Embedding(args.vocabSize, args.dModel);
    this.encoder = new Encoder(args);
    this.decoder = new Decoder(args);
    if (!args.tieWordEmbeddings)
      this.lmHead = new nn.Linear(args.dModel, args.vocabSize, false);

    this.hasEncoder = true;
    this.decoderStartToken = args.decoderStartTokenId;
  }

  override computeTextEmbeddings(inputs: mx.array): mx.array {
    return this.shared.forward(inputs);
  }

  override decodeEmbeddings(embeddings: mx.array, memory: mx.array, cache?: BaseKVCache[]): mx.array {
    if (!memory)
      throw new Error('This model is not decoder-only.');
    const mask = createAttentionMask(embeddings, cache);
    let y = this.decoder.forward(embeddings, memory, mask, undefined, cache);
    if (this.lmHead)
      return this.lmHead.forward(y);
    y = mx.multiply(y, this.args.dModel ** -0.5);
    y = mx.matmul(y, this.shared.weight.T);
    return y;
  }

  override encodeEmbeddings(embeddings: mx.array): mx.array {
    return this.encoder.forward(embeddings);
  }

  override getDecoderKVCacheOptions(): KVCacheOptions {
    return {
      nLayers: this.decoder.block.length,
      headDim: this.args.dKv,
      nKVHeads: this.args.numHeads,
    };
  }
}
