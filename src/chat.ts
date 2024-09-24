#!/usr/bin/env node

import readline from 'node:readline/promises';
import {core as mx} from '@frost-beta/mlx'
import {BaseModel, Message, Tokenizer, loadModel, step} from './llm.js'

if (process.argv.length < 3) {
  console.error('Usage: llm-chat /path/to/weights/dir');
  process.exit(0);
}

main(process.argv[2])

async function main(dir: string) {
  // Load tokenizer.
  const tokenizer = new Tokenizer(dir);

  // Load model.
  const model = await loadModel(dir);

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
    const reply = await talk(tokenizer, model, messages);
    messages.push({role: 'assistant', content: reply});
  }
}

// Send full messages history to model and get response.
async function talk(tokenizer: Tokenizer, model: BaseModel, messages: Message[]) {
  // Translate the messages to tokens.
  const promptTokens = tokenizer.applyChatTemplate(messages);

  // Predict next tokens.
  let tokens: number[] = [];
  let text = '';
  for await (const [ token ] of step(promptTokens, model, tokenizer.eosToken, 0.8)) {
    tokens.push(token);
    const char = tokenizer.decode(tokens);
    // The token may represent an incomplete unicode char.
    if (char.endsWith('\u{FFFD}'))
      continue;
    text += char;
    process.stdout.write(char);
    tokens = [];
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
  return text;
}
