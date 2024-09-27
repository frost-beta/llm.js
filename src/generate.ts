#!/usr/bin/env node

import {ImageProcessor, Tokenizer, loadModel, step} from './llm.js';

let maxTokens = 512;
const argv = process.argv.slice(2).filter((arg) => {
  if (arg.startsWith('--max-tokens=')) {
    maxTokens = parseInt(arg.substring(arg.indexOf('=') + 1));
    return false;
  }
  return true;
})

if (argv.length < 1) {
  console.error('Usage: llm-generate /path/to/weights/dir [--max-tokens=512] [prompt]');
  process.exit(0);
}

main(argv[0], argv[1]);

async function main(dir: string, prompt?: string) {
  const tokenizer = new Tokenizer(dir);
  const model = await loadModel(dir);

  let imageProcessor: ImageProcessor | undefined;
  if (model.imagePlaceholder)
    imageProcessor = new ImageProcessor(dir);

  if (prompt)
    process.stdout.write(prompt);

  // Encode prompt or just use BOS.
  const {bosToken, eosToken} = tokenizer;
  let promptTokens = prompt ? tokenizer.encode(prompt) : [ bosToken ];
  // Some tokenizers append EOS to the encoded text, remove it otherwise the
  // generation might stop there.
  if (promptTokens.length > 1 && promptTokens.at(-1) === eosToken)
    promptTokens = promptTokens.slice(0, -1);

  // Generation.
  let tokens: number[] = [];
  let count = 0;
  for await (const [ token ] of step(promptTokens, model, eosToken)) {
    if (++count > maxTokens)
      break;
    tokens.push(token);
    const char = tokenizer.decode(tokens);
    // The token may represent an incomplete unicode char.
    if (char.endsWith('\u{FFFD}'))
      continue;
    process.stdout.write(char);
    tokens = [];
  }
  process.stdout.write('\n');
}
