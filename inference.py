import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from model import Seq2SeqModel

from utils.data_generator import BatchGenerator
from utils.vocab import Vocab, VocabEntry
import utils.param_provider as param_provider
import utils.misc as util
from utils.misc import CUDA_wrapper, masked_cross_entropy, bleu_score, transform_tensor_to_list_of_snts

import os
import pickle
from time import time


train_params = param_provider.get_train_params()
model_params = param_provider.get_model_params()

for key, val in train_params.items():
    exec(key + '=val')
for key, val in model_params.items():
    exec(key + '=val')

if start_model is None:
    raise ValueError('you should specify a model to infer from')

batch_generator = BatchGenerator(
    vocab_path=vocab_path, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, test_batch_size=test_batch_size
)

if use_masked_loss:
    loss_function = masked_cross_entropy
else:
    loss_function = nn.CrossEntropyLoss()

vocab_size_encoder = len(batch_generator.vocab.src)
vocab_size_decoder = len(batch_generator.vocab.tgt)

technical_params = {
    'vocab_size_encoder': vocab_size_encoder,
    'vocab_size_decoder': vocab_size_decoder,
    'enc_pre_emb': None,
    'dec_pre_emb': None,
    'embed_dim': 300,
    'hidden_size': 256
}

model = CUDA_wrapper(
    Seq2SeqModel(
        **technical_params,
        output_mode=output_mode,
        baseline_output_mode=baseline_output_mode,
        feed_mode=feed_mode, 
        baseline_feed_mode=baseline_feed_mode,
        attention_mode=attention_mode, 
        baseline_attention_mode=baseline_attention_mode
    )
)
print('inferring from model:', start_model)
model.load_state_dict(torch.load(os.path.join('./saved_models', start_model)))

hyp_to_save = []
ref_to_save = []

while True:
    stop_iteration_flag, rf, chunk_batch_torch, tgt_batch_torch, tgt_len = batch_generator.next_test()
    (unscaled_logits, outputs), \
    (unscaled_logits_feed_baseline, outputs_feed_baseline), \
    (unscaled_logits_attn_baseline, outputs_attn_baseline) = model(chunk_batch_torch, tgt_batch_torch, work_mode='inference')
    hypothesis = transform_tensor_to_list_of_snts(outputs, batch_generator.vocab.tgt.id2word)
    reference = [[cur_ref[:-1]] for cur_ref in rf]
    list_of_hypotheses = hypothesis
    list_of_references = reference
    hyp_to_save += list_of_hypotheses
    ref_to_save += list_of_references
    if stop_iteration_flag:
        break

hyp_str = ""
ref_str = ""
for i in hyp_to_save:
    s = ' '.join(i)
    hyp_str += s + '\n'
for i in ref_to_save:
    s = ' '.join(i[0])
    ref_str += s + '\n'
with open('reference', 'w') as f:
    f.write(ref_str)
with open('output', 'w') as f:
    f.write(hyp_str)
