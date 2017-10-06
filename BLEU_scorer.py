import torch
from data_generator import BatchGenerator # data generator for translation
from vocab import Vocab, VocabEntry
from model import Seq2SeqModel, Encoder, Decoder
from util import CUDA_wrapper
from time import time
from torch import optim
import torch.autograd as autograd
from text_reverse_data_generator import get_data_generators, revert_words # data generator for text_reverse
import torch.nn as nn
import datetime
from util import id_to_char, char_to_id, masked_cross_entropy
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import nltk
import fasttext
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 9)

# %% loading the model
vocab_path="data/nmt_iwslt/vocab.bin"
train_batch_size = 80 # that was 256
eval_batch_size = 64
test_batch_size = 64
bg = BatchGenerator(vocab_path=vocab_path, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, test_batch_size=test_batch_size)

task = 'translation'
vocab_size_encoder = len(bg.vocab.src)
vocab_size_decoder = len(bg.vocab.tgt)

with open('data/fasttext/my_de_emb', 'rb') as f:
    de_emb = pickle.load(f)

with open('data/fasttext/my_en_emb', 'rb') as f:
    en_emb = pickle.load(f)
#vocab_size_encoder=vocab_size_encoder, vocab_size_decoder=vocab_size_decoder
model = Seq2SeqModel(vocab_size_encoder=vocab_size_encoder, vocab_size_decoder=vocab_size_decoder, embed_dim=128, hidden_size=256, enc_pre_emb=de_emb, dec_pre_emb=en_emb)
model.load_state_dict(torch.load('saved_model_step=59999_time=2017-10-06 06:07:49.275572'))
# %%
loss_function = masked_cross_entropy

# %%

bleu_scores = []
loss_scores = []
def transform_seq_to_sent(seq, vcb):
    return ' '.join([vcb[i] for i in seq])

def transform_tensor_to_list_of_snts(tensor, vcb):
    np_tens = tensor.data.cpu().numpy()
    snts = []
    end_snt = "</s>"
    for i in np_tens:
        cur_snt = transform_seq_to_sent(i, vcb)
        snts.append(cur_snt[:cur_snt.index("</s>") if end_snt in cur_snt else len(cur_snt)].split())
    return snts

def bleu_score(unscaled_logits, outputs, rev_chunk_batch_torch, vcb_id2word):
    hypothesis = transform_tensor_to_list_of_snts(outputs, vcb_id2word)
    reference = transform_tensor_to_list_of_snts(rev_chunk_batch_torch, vcb_id2word)
    reference = [[cur_ref] for cur_ref in reference]
    list_of_hypotheses = hypothesis
    list_of_references = reference
    return nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)

chunk_batch_torch, rev_chunk_batch_torch, tgt_len = bg.next_eval()
unscaled_logits, outputs = model(chunk_batch_torch, rev_chunk_batch_torch, work_mode='test')
bleu_score(unscaled_logits, outputs, rev_chunk_batch_torch, bg.vocab.tgt.id2word)

#%%
hyp_to_save = []
ref_to_save = []

for _ in range(100):
    chunk_batch_torch, rev_chunk_batch_torch, tgt_len = bg.next_eval()
    rev_chunk_batch_torch.data.shape
    unscaled_logits, outputs = model(chunk_batch_torch, rev_chunk_batch_torch, work_mode='test')
    outputs.data.shape
    rev_chunk_batch_torch.data.shape
    hypothesis = transform_tensor_to_list_of_snts(outputs, bg.vocab.tgt.id2word)
    reference = transform_tensor_to_list_of_snts(rev_chunk_batch_torch, bg.vocab.tgt.id2word)
    reference = [[cur_ref] for cur_ref in reference]
    list_of_hypotheses = hypothesis
    list_of_references = reference
    hyp_to_save += list_of_hypotheses
    ref_to_save += list_of_references
    # bleu_scores.append(nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses))
    # loss_scores.append(loss_function(unscaled_logits, rev_chunk_batch_torch, tgt_len))
# %%
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
#
# # %%
# np.mean(bleu_scores)
# loss_function(unscaled_logits, rev_chunk_batch_torch, tgt_len)
# nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)
# list_of_hypotheses
# rev_chunk_batch_torch.data.shape
# unscaled_logits.data.shape
# np.mean(loss_scores)
# # %%
# unscaled_logits.data.shape
# rev_chunk_batch_torch.data.shape
#
# # %%
# def sequence_mask(sequence_length, max_len=None):
#     if max_len is None:
#         max_len = sequence_length.data.max()
#     batch_size = sequence_length.size(0)
#     seq_range = torch.range(0, max_len - 1).long()
#     seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
#     seq_range_expand = autograd.Variable(seq_range_expand)
#     if sequence_length.is_cuda:
#         seq_range_expand = seq_range_expand.cuda()
#     seq_length_expand = (sequence_length.unsqueeze(1)
#                          .expand_as(seq_range_expand))
#     return seq_range_expand < seq_length_expand
#
# mm = nn.LogSoftmax()
#
# logits = mm(unscaled_logits)
# target = rev_chunk_batch_torch
# length = tgt_len
# length = autograd.Variable(torch.LongTensor(length)).cuda()
#
# # cut len of logits
# logits = logits[:, :target.size()[1], :].contiguous()
#
#
# # logits_flat: (batch * max_len, num_classes)
# logits_flat = logits.view(-1, logits.size(-1))
# # log_probs_flat: (batch * max_len, num_classes)
# log_probs_flat = nn.functional.log_softmax(logits_flat)
# # target_flat: (batch * max_len, 1)
# target_flat = target.view(-1, 1)
# # losses_flat: (batch * max_len, 1)
# log_probs_flat.data.shape
# target_flat.data.shape
# losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
# # losses: (batch, max_len)
# losses = losses_flat.view(*target.size())
# # mask: (batch, max_len)
# mask = sequence_mask(sequence_length=length, max_len=target.size(1))
# print(mask)
# losses = losses * mask.float()
# loss = losses.sum() / length.float().sum()
# loss
# # %%
# # %%
# cum_eval_loss = 0
# cum_eval_acc = 0
# transform_seq_to_sent(outputs_np[0], bg.vocab.tgt.id2word)
#
# outputs_np = outputs.data.cpu().numpy()
# cur_decode_batch_size = 10
# for i in range(cur_decode_batch_size):
#     print('{}\n|  vs  |\n{}'.format(
#         ' '.join([bg.vocab.tgt.id2word[k.data.cpu().numpy()[0]] for k in rev_chunk_batch_torch[i]]),
#         ' '.join([bg.vocab.tgt.id2word[k] for k in outputs_np[i]])
#     ))
#
# # %%
# bleu_scores
# # %% NOT BLEU ###############
# # %%
# print('value')
# embedding = nn.Embedding(10, 3)
# input = autograd.Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]])).transpose(0, 1).contiguous()
# inp_emb = embedding(input)
# inp_emb.data.shape
# packed = torch.nn.utils.rnn.pack_padded_sequence(inp_emb, [4, 3])
# torch.nn.utils.rnn.pad_packed_sequence(packed)[0].transpose(0, 1)
