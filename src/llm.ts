import {core as mx} from '@frost-beta/mlx';
import {BaseModel, loadModel, step} from './base.js';
import {Tokenizer} from './tokenizer.js';
import {ImageProcessor} from './image-processor.js';

/**
 * Options for the LLM.generate method.
 */
export interface LLMGenerateOptions {
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

  async *generate(promptTokens: number[], {maxTokens}: LLMGenerateOptions = {}) {
    // Predict next tokens.
    let buffer: number[] = [];
    for await (const [ token ] of step(promptTokens, this.model, this.tokenizer.eosToken)) {
      buffer.push(token);
      const text = this.tokenizer.decode(buffer);
      // The token may represent an incomplete unicode char.
      if (text.endsWith('\u{FFFD}'))
        continue;
      yield text;
      buffer = [];
    }
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
 * Find out all the <image: path> tags in the text and replace with placeholders
 * of the model.
 */
export async function parseImagesInText(text: string, model: BaseModel) {
  const paths: string[] = [];
  text = text.replace(/<image: (.*?)>/g, (match) => {
    paths.push(match.slice(8, -1));
    return model.imagePlaceholder;
  });
}
