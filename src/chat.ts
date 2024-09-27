#!/usr/bin/env node

import readline from 'node:readline/promises';
import {core as mx} from '@frost-beta/mlx';
import {LLM, parseArgs, loadLLM} from './llm.js';
import {Message} from './tokenizer.js';

const [ argv, options ] = parseArgs(process.argv.slice(2));
if (argv.length < 1) {
  console.error('Usage: llm-chat /path/to/weights/dir [--max-tokens=512]');
  process.exit(0);
}

main(argv[0]);

async function main(dir: string) {
  const llm = await loadLLM(dir);

  // Records the messages.
  const messages: Message[] = [];

  // Chat interface.
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  rl.once('close', () => process.stdout.write('\n'));
  while (true) {
    const question = await rl.question('You> ')
    messages.push({role: 'user', content: question});
    process.stdout.write('Assistant> ');
    const reply = await talk(llm, messages.at(-1), messages.length == 1);
    messages.push({role: 'assistant', content: reply});
  }
}

// Send full messages history to model and get response.
async function talk(llm: LLM, message: Message, firstMessage: boolean) {
  // Translate the messages to tokens.
  // Note that some chat templates add a system prompt automatically and we need
  // to trim it when generating tokens for only new messages.
  const promptTokens = llm.tokenizer.applyChatTemplate([ message ], {trimSystemPrompt: !firstMessage});

  const promptTensor = mx.array(promptTokens, mx.int32).index(mx.newaxis);
  const promptEmbeds = llm.model.computeTextEmbeddings(promptTensor);

  // Predict next tokens.
  let result = '';
  for await (const text of llm.generate(promptEmbeds)) {
    result += text;
    process.stdout.write(text);
  }
  process.stdout.write('\n');

  if (false) {  // used for debugging leaks
    console.log(`MLX RAM ${(mx.metal.getActiveMemory() / 1024 ** 2).toFixed(1)}M,`,
                `Cache ${(mx.metal.getCacheMemory() / 1024 ** 2).toFixed(1)}M,`,
                `JS Objects ${mx.getWrappersCount()}.`);
  }

  if (mx.metal.isAvailable()) {
    // After a conversation, we know it will take a while before next input and
    // it is good chance to just release all the memory cache.
    mx.metal.clearCache();
  }
  return result;
}
