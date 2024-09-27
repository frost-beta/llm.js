#!/usr/bin/env node

import {loadLLM} from './llm.js';

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
  const llm = await loadLLM(dir);

  // Encode prompt or just use BOS.
  const {bosToken, eosToken} = llm.tokenizer;
  let promptTokens = prompt ? llm.tokenizer.encode(prompt) : [ bosToken ];
  // Some tokenizers append EOS to the encoded text, remove it otherwise the
  // generation might stop there.
  if (promptTokens.length > 1 && promptTokens.at(-1) === eosToken)
    promptTokens = promptTokens.slice(0, -1);

  // Generation.
  if (prompt)
    process.stdout.write(prompt);
  for await (const text of llm.generate(promptTokens, {maxTokens}))
    process.stdout.write(text);
  process.stdout.write('\n');
}
