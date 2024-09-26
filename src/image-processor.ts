import sharp from 'sharp';
import {core as mx} from '@frost-beta/mlx';

export type ImageInputType = Buffer | ArrayBuffer | string;

export interface PreprocessorConfig {
  cropSize: number | {width: number, height: number},
  doCenterCrop: boolean,
  doNormalize: boolean,
  doRescale: boolean,
  doResize: boolean,
  imageMean?: number[],
  imageStd?: number[],
  rescaleFactor?: number,
  size: number | {shortestEdge: number},
}

export interface ProcessedImage {
  data: Buffer;
  info: sharp.OutputInfo;
}

export class ClipImageProcessor {
  cropSize: {width: number, height: number};
  shortestEdge: number;

  constructor(private config: PreprocessorConfig) {
    if (!config.cropSize || !config.size)
      throw new Error('"crop_size" and "size" are required.');
    // The config.crop_size can be a number or an object.
    if (typeof config.cropSize == 'number')
      this.cropSize = {width: config.cropSize, height: config.cropSize};
    else
      this.cropSize = config.cropSize;
    // The config.size can be a number or an object.
    if (typeof config.size == 'number')
      this.shortestEdge = config.size;
    else
      this.shortestEdge = config.size.shortestEdge;
  }

  async processImage(input: ImageInputType): Promise<ProcessedImage> {
    let image = sharp(input);
    if (this.config.doResize &&
        this.config.doCenterCrop &&
        ((this.shortestEdge == this.cropSize.width && this.cropSize.width <= this.cropSize.height) ||
         (this.shortestEdge == this.cropSize.height && this.cropSize.height <= this.cropSize.width))) {
      // Fast path for resize and crop with same size.
      image = image.resize(this.cropSize.width, this.cropSize.height);
    } else {
      // Slow path for doing resize and crop in 2 separate steps.
      if (this.config.doResize)
        image = image.resize(this.shortestEdge, this.shortestEdge, {fit: 'outside'});
      if (this.config.doCenterCrop)
        image = await centerCrop(image, this.cropSize);
    }
    // The model only works with RGB.
    image = image.removeAlpha();
    // Extract size and data.
    return await image.raw().toBuffer({resolveWithObject: true});
  }

  processImages(inputs: ImageInputType[]): Promise<ProcessedImage[]> {
    return Promise.all(inputs.map(this.processImage.bind(this)));
  }

  normalizeImages(images: ProcessedImage[]) {
    const {info} = images[0];
    // The model expects the data to be a nested array.
    let tensor = mx.stack(images.map(i => mx.array(Array.from(i.data))));
    tensor = tensor.reshape([ images.length, info.width, info.height, 3 ]);
    // Normalize the tensor.
    if (this.config.doRescale) {
      tensor = mx.multiply(tensor.astype(mx.float32),
                           this.config.rescaleFactor ?? 1 / 255);
    }
    if (this.config.doNormalize) {
      tensor = mx.divide(mx.subtract(tensor, mx.array(this.config.imageMean!)),
                         mx.array(this.config.imageStd!));
    }
    return tensor;
  }
}

async function centerCrop(image: sharp.Sharp, cropSize: {width: number, height: number}) {
  // Have to call toBuffer to get the new size after resize.
  const {info} = await image.toBuffer({resolveWithObject: true});
  return image.extract({
    top: (info.height - cropSize.height) / 2,
    left: (info.width - cropSize.width) / 2,
    width: cropSize.width,
    height: cropSize.height,
  });
}
