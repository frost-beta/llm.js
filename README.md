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

## Supported models

* Llama [3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) / [3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
* Qwen [2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f) / [2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)

Note: Models using data types other than `float32` require GPU support.

## APIs

```typescript
import sharp from 'sharp';
import { core as mx, nn } from '@frost-beta/mlx';

export abstract class BaseModel extends nn.Module {
    abstract forward(inputs: mx.array,
                     cache?: BaseKVCache[]): mx.array;
    forwardWithPixels(inputIds: mx.array,
                      pixelValues: mx.array,
                      cache?: BaseKVCache[]): mx.array;
    sanitize(weights: Record<string, mx.array>): void;
    abstract get layers(): nn.Module[];
    abstract get headDim(): number;
    abstract get nKVHeads(): number;
}

export abstract class BaseKVCache {
    abstract updateAndFetch(keys: mx.array,
                            values: mx.array): [mx.array, mx.array];
}

export class KVCache extends BaseKVCache {
    constructor(headDim: number, nKVHeads: number);
}

export class RotatingKVCache extends BaseKVCache {
    constructor(headDim: number, nKVHeads: number, maxSize = 1024, keep = 4);
}

export async function loadModel(dir: string): Promise<BaseModel>;

export interface StepOptions {
    topP?: number;
    temperature?: number;
}

export async function* step(promptTokens: number[],
                            model: BaseModel,
                            eosToken: number,
                            {
                                topP = 0.8,
                                temperature = 1,
                            }?: StepOptions): AsyncGenerator<[number, number], void>;

export function sample(logits: mx.array,
                       topP?: number,
                       temperature?: number): [mx.array, mx.array];

export function topPSampling(logits: mx.array,
                             topP?: number,
                             temperature?: number): mx.array;

export interface Message {
    role: 'user' | 'assistant';
    content: string;
}

export class Tokenizer {
    bosToken: number;
    eosToken: number;
    constructor(dir: string);
    encode(text: string): number[];
    decode(tokens: number[]): string;
    applyChatTemplate(messages: Message[]): number[];
}

export type ImageInputType = Buffer | ArrayBuffer | string;

export interface PreprocessorConfig {
    cropSize: number | {
        width: number;
        height: number;
    };
    doCenterCrop: boolean;
    doNormalize: boolean;
    doRescale: boolean;
    doResize: boolean;
    imageMean?: number[];
    imageStd?: number[];
    rescaleFactor?: number;
    size: number | {
        shortestEdge: number;
    };
}

export interface ProcessedImage {
    data: Buffer;
    info: sharp.OutputInfo;
}

export declare class ImageProcessor {
    constructor(config: PreprocessorConfig);
    processImage(input: ImageInputType): Promise<ProcessedImage>;
    processImages(inputs: ImageInputType[]): Promise<ProcessedImage[]>;
    normalizeImages(images: ProcessedImage[]): mx.array;
}
```

Check [`chat.ts`](https://github.com/frost-beta/llm.js/blob/main/src/chat.ts)
and [`generate.ts`](https://github.com/frost-beta/llm.js/blob/main/src/generate.ts)
for examples.

## CLI

First download weights with any tool you like:

```sh
npm install -g @frost-beta/huggingface
huggingface download --to weights mlx-community/Meta-Llama-3-8B-Instruct-8bit
```

Then start chating:

```sh
npm install -g @frost-beta/llm
llm-chat ./weights
```

Or do text generation:

```sh
llm-generate ./weights 'Write a short story'
```


## License

MIT
