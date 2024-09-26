import {TokenizerLoader} from '@lenml/tokenizers';
import {readJsonSync} from './fs.js';

/**
 * A message in chat models.
 */
export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Options for chat template.
 */
export interface ChatTemplateOptions {
  trimSystemPrompt?: boolean;
}

/**
 * Wraps the tokenizer of transformers.js.
 */
export class Tokenizer {
  bosToken: number;
  eosToken: number;
  private tokenizer: ReturnType<typeof TokenizerLoader.fromPreTrained>;
  private systemPromptLength?: number;

  constructor(dir: string) {
    this.tokenizer = TokenizerLoader.fromPreTrained({
      tokenizerJSON: readJsonSync(`${dir}/tokenizer.json`),
      tokenizerConfig: readJsonSync(`${dir}/tokenizer_config.json`),
    });
    // Do not strip the heading whitespace as it breaks streaming.
    const {decoders} = this.tokenizer.decoder as any;
    if (decoders?.at(-1)?.config?.type == 'Strip')
      decoders.pop();
    // Get EOS token.
    const {tokens_to_ids} = this.tokenizer.model;
    this.eosToken = tokens_to_ids.get(this.tokenizer.getToken('eos_token'));
    // Some models do not have a BOS token, they use EOS instead.
    this.bosToken = tokens_to_ids.get(this.tokenizer.getToken('bos_token')) ?? this.eosToken;
  }

  encode(text: string) {
    return this.tokenizer.encode(text);
  }

  decode(tokens: number[]) {
    return this.tokenizer.decode(tokens);
  }

  applyChatTemplate(messages: Message[],
                    {
                      trimSystemPrompt = false,
                    }: ChatTemplateOptions = {}): number[] {
    if (trimSystemPrompt && this.systemPromptLength === undefined) {
      // Get the automatically inserted system prompt by passing empty messages.
      const systemPrompt = this.tokenizer.apply_chat_template([], {
        add_generation_prompt: false,
        tools: null,
      } as unknown) as number[];
      this.systemPromptLength = systemPrompt.length;
    }
    const tokens = this.tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      // https://github.com/xenova/transformers.js/issues/879
      tools: null,
    } as unknown) as number[];
    if (trimSystemPrompt)
      return tokens.slice(this.systemPromptLength);
    return tokens;
  }
}
