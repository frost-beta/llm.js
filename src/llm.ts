import {fileURLToPath} from 'node:url';
import {core as mx} from '@frost-beta/mlx';
import {BaseModel, StepOptions, loadModel, step} from './base.js';
import {BaseKVCache, KVCache, RotatingKVCache} from './kv-cache.js';
import {ChatTemplateOptions, Message, Tokenizer} from './tokenizer.js';
import {ImageInputType, ImageProcessor} from './image-processor.js';

/**
 * Options for the LLM.generate method.
 */
export interface LLMGenerateOptions extends StepOptions {
  maxTokens?: number;
  maxKVSize?: number;
}

/**
 * Parse the args for the generate options.
 */
export function parseArgs(args: string[]): [ string[], LLMGenerateOptions ] {
  const options: LLMGenerateOptions = {};
  args = args.filter((arg) => {
    if (arg.startsWith('--max-tokens=')) {
      options.maxTokens = parseInt(arg.substring(arg.indexOf('=') + 1));
      return false;
    }
    if (arg.startsWith('--max-kv-size=')) {
      options.maxKVSize = parseInt(arg.substring(arg.indexOf('=') + 1));
      return false;
    }
    if (arg.startsWith('--temperature=')) {
      options.temperature = parseFloat(arg.substring(arg.indexOf('=') + 1));
      return false;
    }
    if (arg.startsWith('--top-p=')) {
      options.topP = parseFloat(arg.substring(arg.indexOf('=') + 1));
      return false;
    }
    return true;
  });
  return [ args.map(unescapeBackslashes), options ];
}

/**
 * Wraps language models with or without vision.
 */
export class LLM {
  kvCache?: BaseKVCache[];

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
      if (!this.model.hasEncoder && tokens.length > 1 && tokens.at(-1) == this.tokenizer.eosToken)
        tokens.pop();
    } else {
      tokens = [ this.tokenizer.bosToken ];
    }
    const embeddings = this.tokensToEmbeddings(tokens, pixelEmbeds);
    mx.dispose(pixelEmbeds);
    return embeddings;
  }

  /**
   * Convert the messages to embeddings, with images parsed.
   */
  async applyChatTemplate(messages: Message[], options?: ChatTemplateOptions) {
    // Receive the images in all the messages.
    let pixelEmbedsList: mx.array[] = [];
    for (const message of messages) {
      const [ text, pixelEmbeds ] = await this.parseImagesInText(message.content);
      if (pixelEmbeds) {
        message.content = text;
        pixelEmbedsList.push(pixelEmbeds);
      }
    }
    // Create embeddings for the messages and the images.
    const tokens = this.tokenizer.applyChatTemplate(messages, options);
    const pixelEmbeds = pixelEmbedsList.length > 0 ? mx.concatenate(pixelEmbedsList, 0)
                                                   : undefined;
    const embeddings = this.tokensToEmbeddings(tokens, pixelEmbeds);
    mx.dispose(pixelEmbeds, pixelEmbedsList);
    return embeddings;
  }

  /**
   * Predict next tokens using the embeddings of prompt.
   */
  async *generate(promptEmbeds: mx.array, options: LLMGenerateOptions = {}) {
    const [ batchSize ] = promptEmbeds.shape;
    // If not specified, create a shared cache between generations.
    if (!options.kvCache) {
      if (!this.kvCache) {
        const kvCacheOptions = this.model.getDecoderKVCacheOptions();
        if (options.maxKVSize)
          this.kvCache = RotatingKVCache.create(kvCacheOptions, options.maxKVSize);
        else
          this.kvCache = KVCache.create(kvCacheOptions);
      }
      options.kvCache = this.kvCache;
    }
    // Predict next tokens.
    let buffers: number[][] = Array.from({length: batchSize}, () => []);
    let count = 0;
    for await (const tokens of step(promptEmbeds, this.model, this.tokenizer.eosToken, options)) {
      ++count;
      if (options.maxTokens && count > options.maxTokens)
        break;
      const results: string[] = Array.from({length: batchSize}, () => '');
      for (let i = 0; i < batchSize; ++ i) {
        buffers[i].push(tokens[i]);
        let text = this.tokenizer.decode(buffers[i]);
        // The token may represent an incomplete unicode char.
        if (text.endsWith('\u{FFFD}'))
          continue;
        // Trim left whitespace for the first output.
        if (this.tokenizer.trimLeft && count == 1)
          text = text.trimLeft();
        results[i] = text;
        buffers[i] = [];
      }
      yield results;
    }
  }

  /**
   * Find out all the <image: path> tags in the text and replace them with
   * placeholders of the model.
   */
  private async parseImagesInText(text: string): Promise<[ string, mx.array | undefined ]> {
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
    return mx.tidy(() => {
      const pixels = this.imageProcessor.normalizeImages(images);
      return [ text, this.model.computePixelEmbeddings(pixels) ];
    });
  }

  /**
   * Convert tokens and images to embeddings.
   */
  private tokensToEmbeddings(tokens: number[], pixelEmbeds?: mx.array) {
    return mx.tidy(() => {
      const inputs = mx.array(tokens, mx.int32).index(mx.newaxis);
      const inputEmbeds = this.model.computeTextEmbeddings(inputs);
      if (!pixelEmbeds)
        return inputEmbeds;
      return this.model.mergeTextPixelEmbeddings(inputs, inputEmbeds, pixelEmbeds);
    });
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

// Get image from the path.
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

// Unescape backslashes in string.
function unescapeBackslashes(str: string): string {
  return str.replace(/\\(.)/g, function(_, escapeChar) {
    switch (escapeChar) {
      case 'n':
        return '\n';
      case 'r':
        return '\r';
      case 't':
        return '\t';
      default:
        return escapeChar;
    }
  });
}
