import {core as mx, nn} from '@frost-beta/mlx';
import {baseModelArgs} from '../../base.js';

export interface VisionConfig {
  modelType: 'clip_vision_model';
  numHiddenLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  imageSize: number;
  patchSize: number;
  projectionDim: number;
  vocabSize: number;
  numChannels: number;
  layerNormEps: number;
}

class Attention extends nn.Module {
  numHeads: number;
  qProj: nn.Linear;
  kProj: nn.Linear;
  vProj: nn.Linear;
  outProj: nn.Linear;

  constructor(dims: number, numHeads: number, bias = false) {
    super();
    this.numHeads = numHeads;

    this.qProj = new nn.Linear(dims, dims, bias);
    this.kProj = new nn.Linear(dims, dims, bias);
    this.vProj = new nn.Linear(dims, dims, bias);
    this.outProj = new nn.Linear(dims, dims, bias);
  }

  forward(queries: mx.array, keys: mx.array, values: mx.array, mask?: mx.array) {
    queries = this.qProj.forward(queries);
    keys = this.kProj.forward(keys);
    values = this.vProj.forward(values);

    const [ B, L, D ] = queries.shape;
    const [  , S,   ] = keys.shape;

    queries = queries.reshape(B, L, this.numHeads, -1).transpose(0, 2, 1, 3);
    keys = keys.reshape(B, S, this.numHeads, -1).transpose(0, 2, 3, 1);
    values = values.reshape(B, S, this.numHeads, -1).transpose(0, 2, 1, 3);

    const scale = Math.sqrt(1 / queries.shape.at(-1));
    let scores = mx.matmul(mx.multiply(queries, scale), keys);
    if (mask)
      scores = mx.add(scores, mask.astype(scores.dtype));
    scores = mx.softmax(scores, -1);
    const valuesHat = mx.matmul(scores, values).transpose(0, 2, 1, 3).reshape(B, L, -1);

    return this.outProj.forward(valuesHat);
  }
}

class MLP extends nn.Module {
  activationFn = new nn.GELU('fast');
  fc1: nn.Linear;
  fc2: nn.Linear;

  constructor(config: VisionConfig) {
    super();
    this.fc1 = new nn.Linear(config.hiddenSize, config.intermediateSize);
    this.fc2 = new nn.Linear(config.intermediateSize, config.hiddenSize);
  }

  forward(x: mx.array) {
    x = this.activationFn.forward(this.fc1.forward(x));
    x = this.fc2.forward(x);
    return x;
  }
}

class EncoderLayer extends nn.Module {
  embedDim: number;
  selfAttn: Attention;
  layerNorm1: nn.LayerNorm;
  mlp: MLP;
  layerNorm2: nn.LayerNorm;

  constructor(config: VisionConfig) {
    super();
    this.embedDim = config.hiddenSize;
    this.selfAttn = new Attention(config.hiddenSize, config.numAttentionHeads, true);
    this.layerNorm1 = new nn.LayerNorm(this.embedDim, config.layerNormEps);
    this.mlp = new MLP(config);
    this.layerNorm2 = new nn.LayerNorm(this.embedDim, config.layerNormEps);
  }

  forward(x: mx.array, mask?: mx.array) {
    let y = this.layerNorm1.forward(x);
    y = this.selfAttn.forward(y, y, y, mask);
    x = mx.add(x, y);
    y = this.layerNorm2.forward(x);
    y = this.mlp.forward(y);
    return mx.add(x, y);
  }
}

class Encoder extends nn.Module {
  layers: EncoderLayer[] = [];

  constructor(config: VisionConfig) {
    super();
    for (let i = 0; i < config.numHiddenLayers; ++i)
      this.layers.push(new EncoderLayer(config));
  }

  forward(x: mx.array, mask?: mx.array) {
    for (let i in this.layers)
      x = this.layers[i].forward(x, mask);
    return x;
  }
}

class VisionEmbeddings extends nn.Module {
  embedDim: number;
  imageSize: number;
  patchSize: number;
  classEmbedding: mx.array;
  patchEmbedding: nn.Conv2d;
  numPatches: number;
  numPositions: number;
  positionEmbedding: nn.Embedding;

  #positionIds: mx.array;

  constructor(config: VisionConfig) {
    super();
    this.embedDim = config.hiddenSize;
    this.imageSize = config.imageSize;
    this.patchSize = config.patchSize;

    this.classEmbedding = mx.zeros([ config.hiddenSize ]);

    this.patchEmbedding = new nn.Conv2d(config.numChannels,
                                        this.embedDim,
                                        this.patchSize,
                                        this.patchSize,
                                        undefined,
                                        undefined,
                                        false);

    this.numPatches = (this.imageSize / this.patchSize) ** 2;
    this.numPositions = this.numPatches + 1;
    this.positionEmbedding = new nn.Embedding(this.numPositions, this.embedDim);

    this.#positionIds = mx.arange(this.numPositions, mx.int32).index(mx.newaxis);
  }

  forward(x: mx.array) {
    const batchSize = x.shape[0];
    let patchEmbeddings = this.patchEmbedding.forward(x);
    patchEmbeddings = mx.flatten(patchEmbeddings, 1, 2);
    const embedDim = patchEmbeddings.shape.at(-1);
    const clsEmbeddings = mx.broadcastTo(this.classEmbedding, [ batchSize, 1, embedDim ]);
    let embeddings = mx.concatenate([ clsEmbeddings, patchEmbeddings ], 1);
    embeddings = mx.add(embeddings, this.positionEmbedding.forward(this.#positionIds));
    return embeddings;
  }
}

class ClipVisionModel extends nn.Module {
  embeddings: VisionEmbeddings;
  preLayrnorm: nn.LayerNorm;
  encoder: Encoder;
  postLayernorm: nn.LayerNorm;

  constructor(config: VisionConfig) {
    super();
    this.embeddings = new VisionEmbeddings(config);
    this.preLayrnorm = new nn.LayerNorm(config.hiddenSize);
    this.encoder = new Encoder(config);
    this.postLayernorm = new nn.LayerNorm(config.hiddenSize);
  }

  forward(x: mx.array, outputHiddenStates = false): [ mx.array, mx.array, mx.array[] ] {
    x = this.embeddings.forward(x);
    x = this.preLayrnorm.forward(x);

    let encoderStates = outputHiddenStates ? [ x ] : null;

    for (const layer of this.encoder.layers) {
      x = layer.forward(x);
      if (outputHiddenStates)
        encoderStates.push(x);
    }

    const poolerOutput = this.postLayernorm.forward(x.index(mx.Slice(), 0, mx.Slice()));
    return [ poolerOutput, x, encoderStates ];
  }
}

export class VisionModel extends nn.Module {
  visionModel: ClipVisionModel;

  constructor(config: VisionConfig) {
    super();
    if (config.modelType != 'clip_vision_model')
      throw new Error(`Unsupported vision model type: ${config.modelType}`);
    this.visionModel = new ClipVisionModel(config);
  }

  forward(x: mx.array, outputHiddenStates = false) {
    return this.visionModel.forward(x, outputHiddenStates);
  }
}
