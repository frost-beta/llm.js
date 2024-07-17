# llm.js

Load language models locally with JavaScript, using
[node-mlx](https://github.com/frost-beta/node-mlx), code modified from
[mlx-examples](https://github.com/ml-explore/mlx-examples).

__Quantized models can only run on Macs with Apple Silicon.__

## Usage

Download weights
(more can be found at [mlx-community](https://huggingface.co/collections/mlx-community/)):

```sh
npm install -g @frost-beta/huggingface
huggingface download --to weights mlx-community/Meta-Llama-3-8B-Instruct-8bit
```

Start chating:

```sh
npm install -g @frost-beta/llm
llm-chat ./weights
```

Or do text generation:

```sh
llm-generate ./weights 'Write a short story'
```
