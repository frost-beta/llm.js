import {core as mx, nn} from '@frost-beta/mlx';
import {BaseModel, baseModelArgs} from '../base.js';
import {BaseKVCache} from '../kv-cache.js';
import {VisionConfig, VisionModel} from './llava/vision.js';
import * as llama from './llama.js';

export interface ModelArgs {
  textConfig: llama.ModelArgs;
  visionConfig: VisionConfig;
  ignoreIndex: number;
  imageTokenIndex: number;
  visionFeatureSelectStrategy: string;
  visionFeatureLayer: number;
  vocabSize: number;
}

export function modelArgs(json: any): ModelArgs {
  const args = baseModelArgs(json);
  args.textConfig = Object.assign({
    hiddenSize: 4096,
    numHiddenLayers: 32,
    intermediateSize: 11008,
    numAttentionHeads: 32,
    rmsNormEps: 1e-6,
    vocabSize: 32000,
    ropeTheta: 10000,
    tieWordEmbeddings: false,
  }, args.textConfig);
  args.visionConfig = Object.assign({
    numHiddenLayers: 24,
    hiddenSize: 1024,
    intermediateSize: 4096,
    numAttentionHeads: 16,
    imageSize: 335,
    patchSize: 14,
    projectionDim: 768,
    vocabSize: 3200,
    numChannels: 3,
    layerNormEps: 1e-5,
  }, args.visionConfig);
  return args;
}

class LlavaMultiModalProjector extends nn.Module {
  linear_1: nn.Linear;
  gelu: nn.GELU;
  linear_2: nn.Linear;

  constructor(args: ModelArgs) {
    super();
    this.linear_1 = new nn.Linear(args.visionConfig.hiddenSize, args.textConfig.hiddenSize, true);
    this.gelu = new nn.GELU();
    this.linear_2 = new nn.Linear(args.textConfig.hiddenSize, args.textConfig.hiddenSize, true);
  }

  forward(x: mx.array) {
    x = this.linear_1.forward(x);
    x = this.gelu.forward(x);
    return this.linear_2.forward(x);
  }
}

export class Model extends BaseModel {
  visionTower: VisionModel;
  languageModel: llama.Model;
  multiModalProjector: LlavaMultiModalProjector;
  visionFeatureLayer: number;
  visionFeatureSelectStrategy: string;

  constructor(json: any) {
    super();
    const args = modelArgs(json);
    this.visionTower = new VisionModel(args.visionConfig);
    this.languageModel = new llama.Model(args.textConfig);
    this.multiModalProjector = new LlavaMultiModalProjector(args);
    this.visionFeatureLayer = args.visionFeatureLayer;
    this.visionFeatureSelectStrategy = args.visionFeatureSelectStrategy;

    this.hasEncoder = this.languageModel.hasEncoder;
    this.imagePlaceholder = '<image>';
    this.imageToken = args.imageTokenIndex;
  }

  override computePixelEmbeddings(pixels: mx.array): mx.array {
    // Get the ouptut hidden states from the vision model.
    const [ , , hiddenStates ] = this.visionTower.forward(pixels, true);

    // Select the hidden states from the desired layer.
    let imageFeatures = hiddenStates.at(this.visionFeatureLayer);
    if (this.visionFeatureSelectStrategy == 'default')
      imageFeatures = imageFeatures.index(mx.Slice(), mx.Slice(1));
    else if (this.visionFeatureSelectStrategy == 'full')
      imageFeatures = imageFeatures;
    else
      throw new Error(`Unexpected feature selection strategy: ${this.visionFeatureSelectStrategy}`);

    // Pass image features through the multi-modal projector.
    return this.multiModalProjector.forward(imageFeatures);
  }

  override computeTextEmbeddings(inputs: mx.array): mx.array {
    return this.languageModel.computeTextEmbeddings(inputs);
  }

  override decodeEmbeddings(embeddings: mx.array, memory?: mx.array, cache?: BaseKVCache[]): mx.array {
    return this.languageModel.decodeEmbeddings(embeddings, memory, cache);
  }

  override sanitize(weights: Record<string, mx.array>) {
    for (const key in weights) {
      // PyTorch Conv2d expects the weight tensor to be of shape:
      // [out_channels, in_channels, kH, KW]
      // MLX Conv2d expects the weight tensor to be of shape:
      // [out_channels, kH, KW, in_channels]
      if (key.endsWith('patch_embedding.weight')) {
        // Some mlx-community models already transposed it for us.
        const {shape} = weights[key];
        if (shape[1] != shape[2])
          weights[key] = weights[key].transpose(0, 2, 3, 1);
      }
    }
  }

  override get layers() {
    return this.languageModel.layers as nn.Module[];
  }

  override get headDim() {
    return this.languageModel.headDim;
  }

  override get nKVHeads() {
    return this.languageModel.nKVHeads;
  }
}
