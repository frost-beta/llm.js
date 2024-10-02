#!/usr/bin/env node

import readline from 'node:readline/promises';
import {styleText} from 'node:util';
import {core as mx} from '@frost-beta/mlx';
import {LLMGenerateOptions, LLM, parseArgs, loadLLM} from './llm.js';
import {Message} from './tokenizer.js';

const [ argv, options ] = parseArgs(process.argv.slice(2));
if (argv.length < 1) {
  console.error('Usage: llm-chat /path/to/weights/dir [--max-tokens=512]');
  process.exit(0);
}

main(argv[0], options);

async function main(dir: string, options: LLMGenerateOptions) {
  const llm = await loadLLM(dir);

  // Records the messages.
  const messages: Message[] = [];

  // Whether to use colors output.
  let youPrompt = 'You> ';
  let botPrompt = 'Assistant> ';
  if (process.stdout.isTTY && process.stdout.hasColors()) {
    youPrompt = styleText('green', youPrompt);
    botPrompt = styleText('blue', botPrompt);
  }

  // Chat interface.
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  rl.once('close', () => process.stdout.write('\n'));

  // Chat loop.
  while (!process.stdin.closed) {
    const question = await rl.question(youPrompt);
    messages.push({role: 'user', content: question});
    process.stdout.write(botPrompt);
    const reply = await talk(rl, llm, messages.at(-1), messages.length == 1, options);
    messages.push({role: 'assistant', content: reply});
  }
}

// Send full messages history to model and get response.
async function talk(rl: readline.Interface,
                    llm: LLM,
                    message: Message,
                    firstMessage: boolean,
                    options: LLMGenerateOptions) {
  // Interrupt generation when Ctrl-C is pressed.
  const controller = new AbortController();
  options.signal = controller.signal;
  const abort = () => controller.abort();
  rl.on('SIGINT', abort);

  // Translate the messages to tokens.
  const promptEmbeds = await llm.applyChatTemplate([ message ], {
    // Some chat templates add a system prompt automatically and we need to trim
    // it when generating tokens for only new messages.
    trimSystemPrompt: !firstMessage,
  });

  // Predict next tokens.
  let result = '';
  for await (const text of llm.generate(promptEmbeds, options)) {
    result += text;
    process.stdout.write(text);
  }
  process.stdout.write('\n');

  if (false) {  // used for debugging leaks
    console.log(`MLX RAM ${(mx.metal.getActiveMemory() / 1024 ** 2).toFixed(1)}M,`,
                `Cache ${(mx.metal.getCacheMemory() / 1024 ** 2).toFixed(1)}M,`,
                `JS Objects ${mx.getWrappersCount()}.`);
  }

  // Cleanup.
  mx.dispose(promptEmbeds);
  if (mx.metal.isAvailable()) {
    // After a conversation, we know it will take a while before next input and
    // it is good chance to just release all the memory cache.
    mx.metal.clearCache();
  }
  rl.removeListener('SIGINT', abort);
  return result;
}
