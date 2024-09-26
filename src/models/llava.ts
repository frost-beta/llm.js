import {core as mx, nn} from '@frost-beta/mlx';
import {BaseModel, BaseKVCache, baseModelArgs} from '../llm.js';
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
  imageTokenIndex: number;
  visionFeatureLayer: number;
  visionFeatureSelectStrategy: string;

  constructor(args: ModelArgs) {
    super();
    args = modelArgs(args);
    this.visionTower = new VisionModel(args.visionConfig);
    this.languageModel = new llama.Model(args.textConfig);
    this.multiModalProjector = new LlavaMultiModalProjector(args);
    this.imageTokenIndex = args.imageTokenIndex;
    this.visionFeatureLayer = args.visionFeatureLayer;
    this.visionFeatureSelectStrategy = args.visionFeatureSelectStrategy;
  }

  getInputEmbeddings(inputIds?: mx.array, pixelValues?: mx.array) {
    // Get the input embeddings from the language model.
    const inputsEmbeds = this.languageModel.model.embedTokens.forward(inputIds);
    if (!pixelValues)
      return inputsEmbeds;

    // Get the ouptut hidden states from the vision model.
    const [ , , hiddenStates ] = this.visionTower.forward(pixelValues.transpose(0, 2, 3, 1), true);

    // Select the hidden states from the desired layer.
    let selectedImageFeature = hiddenStates[this.visionFeatureLayer];
    if (this.visionFeatureSelectStrategy == 'default')
      selectedImageFeature = selectedImageFeature.index(mx.Slice(), mx.Slice(1));
    else if (this.visionFeatureSelectStrategy == 'full')
      selectedImageFeature = selectedImageFeature;
    else
      throw new Error(`Unexpected feature selection strategy: ${this.visionFeatureSelectStrategy}`);

    // Pass image features through the multi-modal projector.
    const imageFeatures = this.multiModalProjector.forward(selectedImageFeature);

    // Insert special image tokens in the inputIds
    return this.mergeInputIdsWithImageFeatures(imageFeatures, inputsEmbeds, inputIds);
  }

  forward(inputs: mx.array, cache?: BaseKVCache[]) {
    return this.languageModel.forward(inputs, cache);
  }

  forwardWithPixels(inputIds: mx.array, pixelValues: mx.array, cache?: BaseKVCache[]) {
    const inputEmbddings = this.getInputEmbeddings(inputIds, pixelValues);
    return this.languageModel.forward(inputIds, cache, inputEmbddings);
  }

  sanitize(weights: Record<string, mx.array>) {
    for (const key in weights) {
      // Remove unused position_ids.
      if (key.includes('position_ids'))
        continue;
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

  get layers() {
    return this.languageModel.layers as nn.Module[];
  }

  get headDim() {
    return this.languageModel.headDim;
  }

  get nKVHeads() {
    return this.languageModel.nKVHeads;
  }

  private mergeInputIdsWithImageFeatures(imageFeatures: mx.array, inputsEmbeds: mx.array, inputIds: mx.array) {
    const [ numImages, numImagePatches, embedDim ] = imageFeatures.shape;

    // Positions of <image> tokens in inputIds, assuming batch size is 1.
    const imagePositions: number[] = [];
    const inputs = inputIds.index(0).tolist() as number[];
    for (let i = 0; i < inputs.length; ++i) {
      if (inputs[i] == this.imageTokenIndex)
        imagePositions.push(i);
    }

    if (imagePositions.length !== numImages) {
      throw new Error(`The number of image tokens (${imagePositions.length}) does not match the number of image inputs (${numImages}).`);
    }

    const textSegments: mx.array[] = [];
    let startIdx = 0;

    for (const position of imagePositions) {
      textSegments.push(inputsEmbeds.index(mx.Slice(), mx.Slice(startIdx, position)));
      startIdx = position + 1;
    }

    const imageEmbeddings = mx.split(imageFeatures, numImages);
    const finalEmbeddings: mx.array[] = [];
    for (let i = 0; i < textSegments.length; ++i) {
      finalEmbeddings.push(textSegments[i], imageEmbeddings[i]);
    }
    finalEmbeddings.push(inputsEmbeds.index(mx.Slice(), mx.Slice(startIdx)));

    // Create a final embedding of shape
    // (1, numImagePatches * numImages + sequenceLen, embedDim)
    return mx.concatenate(finalEmbeddings, 1);
  }
}
