import {core as mx} from '@frost-beta/mlx';
import {readFileSync, readdirSync} from 'node:fs';

// Helper for loading weights from a directory.
export function loadWeights(dir: string) {
  const weights: Record<string, mx.array> = {};
  for (const filename of readdirSync(dir)) {
    if (filename.endsWith('.safetensors'))
      Object.assign(weights, mx.load(`${dir}/${filename}`));
  }
  return weights;
}

// Helper for reading a .json file.
export function readJsonSync(path: string) {
  return JSON.parse(String(readFileSync(path)));
}
