import * as llama from './llama.js';

export class Model extends llama.Model {
  constructor(json: any) {
    super(Object.assign({
      attention_bias: true,
      attention_out_projection_bias: false,
      rope_theta: 1000000,
    }, json));
  }
}
