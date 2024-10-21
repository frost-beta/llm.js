import * as llama from './llama.js';

export class Model extends llama.Model {
  constructor(json: any) {
    if (json.force_use_exact_gelu)
      json.hidden_act = 'gelu';
    else if (!json.hidden_act || json.hidden_act == 'gelu_pytorch_tanh')
      json.hidden_act = 'geluApprox';
    super(json);
  }
}
