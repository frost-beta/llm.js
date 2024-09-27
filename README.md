# llm.js

Node.js module providing inference APIs for large language models, with simple
CLI.

Powered by [node-mlx](https://github.com/frost-beta/node-mlx), a machine
learning framework for Node.js.

## Supported platforms

GPU support:

* Macs with Apple Silicon

CPU support:

* x64 Macs
* x64/arm64 Linux

Note: Models using data types other than `float32` require GPU support.

## Supported models

* Llama [3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) / [3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) / [3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
* Qwen [2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f) / [2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
* LLaVa [1.5](https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0)

You can also find quantized versions of the models at
[MLX Community](https://huggingface.co/mlx-community).

## APIs

```typescript
import { core as mx, nn } from '@frost-beta/mlx';

/**
 * Wraps language models with or without vision.
 */
export class LLM {
    /**
     * Encode text with images into embeddings.
     */
    async encode(text?: string): Promise<mx.array>;
    /**
     * Convert the messages to embeddings, with images parsed.
     */
    async applyChatTemplate(messages: Message[], options?: ChatTemplateOptions): Promise<mx.array>;
    /**
     * Predict next tokens using the embeddings of prompt.
     */
    async *generate(promptEmbeds: mx.array, options?: LLMGenerateOptions): AsyncGenerator<string, void, unknown>;
}

/**
 * Create a LLM instance by loading from directory.
 */
export async function loadLLM(dir: string): Promise<LLM>;

/**
 * Options for chat template.
 */
export interface ChatTemplateOptions {
    trimSystemPrompt?: boolean;
}

/**
 * Options for the LLM.generate method.
 */
export interface LLMGenerateOptions {
    maxTokens?: number;
    topP?: number;
    temperature?: number;
}
```

Check [`chat.ts`](https://github.com/frost-beta/llm.js/blob/main/src/chat.ts)
and [`generate.ts`](https://github.com/frost-beta/llm.js/blob/main/src/generate.ts)
for examples.

## CLI

First download weights with any tool you like:

```console
$ npm install -g @frost-beta/huggingface
$ huggingface download --to weights mlx-community/Meta-Llama-3-8B-Instruct-8bit
```

Then start chating:

```console
$ npm install -g @frost-beta/llm
$ llm-chat ./weights
You> Who are you?
Assistant> I am Qwen, a large language model created by Alibaba Cloud.
```

Or do text generation:

```console
$ llm-generate ./weights 'Write a short story'
In a small village, there lived a girl named Eliza.
```

For vision models, put images in the format of `<image:pathOrUrl>`:

```console
$ huggingface download mlx-community/llava-1.5-7b-4bit
$ llm-chat llava-1.5-7b-4bit --temperature=0
You> What is in this image? <image:https://www.techno-edge.net/imgs/zoom/20089.jpg>
Assistant> The image features a man wearing glasses, holding a book in his hands.
```

## License

MIT
