#!/usr/bin/env node

import {core as mx} from '@frost-beta/mlx';
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
  const promptEmbeds = await llm.encode(prompt);

  if (prompt)
    process.stdout.write(prompt);
  for await (const text of llm.generate(promptEmbeds, {maxTokens}))
    process.stdout.write(text);
  process.stdout.write('\n');
}
