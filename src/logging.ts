import {core as mx} from '@frost-beta/mlx';

export function printGenerationLog() {
  if (process.env.LLM_DEBUG) {
    console.log(`Peak memory ${humanReadableSize(mx.metal.getPeakMemory())},`,
                `Cache memory ${humanReadableSize(mx.metal.getCacheMemory())},`,
                `Native JS objects ${mx.getWrappersCount()}.`);
  }
}

export function humanReadableSize(bytes: number) {
  if (bytes == 0)
    return '0';
  const sizes: string[] = [ 'B', 'KB', 'MB', 'GB', 'TB' ];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return parseFloat((bytes / Math.pow(1024, i)).toFixed(2)) + ' ' + sizes[i];
}
