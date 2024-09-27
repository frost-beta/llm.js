import {fileURLToPath} from 'node:url';
import {core as mx} from '@frost-beta/mlx';
import {BaseModel, StepOptions, loadModel, step} from './base.js';
import {Tokenizer} from './tokenizer.js';
import {ImageInputType, ImageProcessor} from './image-processor.js';

/**
 * Options for the LLM.generate method.
 */
export interface LLMGenerateOptions extends StepOptions {
  maxTokens?: number;
}

/**
 * Wraps language models with or without vision.
 */
export class LLM {
  constructor(public model: BaseModel,
              public tokenizer: Tokenizer,
              public imageProcessor?: ImageProcessor) {
  }

  /**
   * Encode text with images into embeddings.
   */
  async encode(text?: string) {
    let tokens: number[];
    let pixelEmbeds: mx.array | undefined;
    if (text) {
      // Text to tokens.
      [ text, pixelEmbeds ] = await this.parseImagesInText(text);
      tokens = this.tokenizer.encode(text);
      // Some tokenizers append EOS to the encoded text, remove it otherwise the
      // generation might stop there.
      if (tokens.length > 1 && tokens.at(-1) == this.tokenizer.eosToken)
        tokens.pop();
    } else {
      tokens = [ this.tokenizer.bosToken ];
    }
    // Tokens to embeddings.
    const inputs = mx.array(tokens, mx.int32).index(mx.newaxis);
    const inputEmbeds = this.model.computeTextEmbeddings(inputs);
    if (!pixelEmbeds)
      return inputEmbeds;
    return this.model.mergeTextPixelEmbeddings(inputs, inputEmbeds, pixelEmbeds);
  }

  /**
   * Predict next tokens using the embeddings of prompt.
   */
  async *generate(promptEmbeds: mx.array, options: LLMGenerateOptions = {}) {
    // Predict next tokens.
    let buffer: number[] = [];
    let count = 0;
    for await (const token of step(promptEmbeds, this.model, this.tokenizer.eosToken, options)) {
      if (options.maxTokens && ++count > options.maxTokens)
        break;
      buffer.push(token);
      const text = this.tokenizer.decode(buffer);
      // The token may represent an incomplete unicode char.
      if (text.endsWith('\u{FFFD}'))
        continue;
      yield text;
      buffer = [];
    }
  }

  /**
   * Find out all the <image: path> tags in the text and replace them with
   * placeholders of the model.
   */
  async parseImagesInText(text: string): Promise<[ string, mx.array | undefined ]> {
    if (!this.imageProcessor)
      return [ text, undefined ];
    // Find out the tags and replace them.
    const paths: string[] = [];
    text = text.replace(/<image:(.*?)>/g, (match) => {
      paths.push(match.slice(7, -1).trim());
      return this.model.imagePlaceholder;
    });
    if (paths.length == 0)
      return [ text, undefined ];
    // Read and process the images.
    const inputs = await Promise.all(paths.map(fetchImage));
    const images = await this.imageProcessor.processImages(inputs);
    const pixels = this.imageProcessor.normalizeImages(images);
    return [ text, this.model.computePixelEmbeddings(pixels) ];
  }
}

/**
 * Create a LLM instance by loading from directory.
 */
export async function loadLLM(dir: string) {
  const model = await loadModel(dir);
  return new LLM(model,
                 new Tokenizer(dir),
                 model.imagePlaceholder ? new ImageProcessor(dir) : undefined);
}

/**
 * Get image from the path.
 */
async function fetchImage(path: string): Promise<ImageInputType> {
  try {
    // Parse url path.
    const url = new URL(path);
    if (url.protocol == 'file:') {
      return fileURLToPath(url);
    } else {
      const response = await fetch(url);
      return await response.arrayBuffer();
    }
  } catch (error) {
    if (error instanceof TypeError && error.message == 'Invalid URL') {
      // If it is not url then treat it as path.
      return path;
    } else {
      throw new Error(`Can not get image from the query URL: ${error.message}`);
    }
  }
}
