#!/usr/bin/env node

import {core as mx} from '@frost-beta/mlx';
import {parseArgs, loadLLM} from './llm.js';

const [ argv, options ] = parseArgs(process.argv.slice(2));
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
  for await (const text of llm.generate(promptEmbeds, options))
    process.stdout.write(text);
  process.stdout.write('\n');
}
