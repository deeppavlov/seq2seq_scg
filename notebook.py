# from matplotlib import pyplot as plt
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
# plt.rcParams['figure.figsize'] = (15, 9)
import numpy as np
import torch
from data_generator import BatchGenerator # data generator for translation
from vocab import Vocab, VocabEntry
from model import Seq2SeqModel, Encoder, Decoder
from util import CUDA_wrapper
from time import time
from torch import optim
import torch.autograd as autograd
#from text_reverse_data_generator import get_data_generators, revert_words # data generator for text_reverse
import torch.nn as nn
import torch.nn.functional as F
import datetime
from util import id_to_char, char_to_id, masked_cross_entropy
import pickle
import nltk
# %%  <- this symbols mean "cell seporator" in Hydrogen plugin for Atom
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
# %%
# src, tgt = bg.next()
# bg.vocab.tgt.id2word[2]
# tgt[0]
#
# # %%
# enc = Encoder(len(bg.vocab.src), 4, 5)
# dec = Decoder(len(bg.vocab.tgt), enc.embedding, 4, 5)
# a, b = bg.next()
# a.data.shape
# b.data.shape
# all_hidden, all_cell = enc(a)
# dec(all_hidden, all_cell, b, enc.batch_size)
# %%

train_batch_size = 128 # that was 200
eval_batch_size = 64
test_batch_size = 64
decode_batch_size = 8

vocab_path="data/nmt_iwslt/vocab.bin"
bg = BatchGenerator(vocab_path=vocab_path, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, test_batch_size=test_batch_size)
task = 'translation'

chunk_length = 32

train_losses_av_mean = {}
train_accs_av_mean = {}

train_losses_pav_av_mean = {}
train_accs_pav_av_mean = {}

#eval_losses_av_mean = {}
#eval_accs_av_mean = {}

train_losses_pav_av_std = {}
train_accs_pav_av_std = {}

train_losses_av_std = {}
train_accs_av_std = {}

use_masked_loss = True
if use_masked_loss:
    loss_function = masked_cross_entropy
else:
    loss_function = nn.CrossEntropyLoss()

vocab_size_encoder = len(bg.vocab.src)
vocab_size_decoder = len(bg.vocab.tgt)

num_runs = 1

#num_steps = 60000
#print_skip = 50
#save_per_step = 10000
num_steps = 5000
print_skip = 100
save_per_step = 1000

train_losses = []
train_accs = []

eval_losses = []
eval_accs = []

NOT_AVAILABLE = 'NA'

feed_mode = 'gumbel-st'
tf_ratio_range = (0.0, 0.0)
feed_baseline = 'argmax'
if feed_mode != 'sampling':
    feed_baseline = NOT_AVAILABLE
softmax_t_range = (1.0, 0.01)
if feed_mode in ['argmax', 'sampling']:
    softmax_t_range = NOT_AVAILABLE
baseline_feed_mode = None
if feed_baseline == 'argmax':
    baseline_feed_mode = 'argmax'
attention_mode = 'hard'
attention_baseline = 'argmax'
if attention_mode != 'hard':
    attention_baseline = NOT_AVAILABLE
baseline_attention_mode = None
if attention_baseline == 'argmax':
    baseline_attention_mode = 'argmax'

mode_name = 'feed=' + feed_mode + '__tf_ratio=' + str(tf_ratio_range) + '__softmax_t=' + str(softmax_t_range) + '__feed_baseline=' + feed_baseline + '__attn=' + attention_mode + '__attn_baseline=' + attention_baseline

do_eval = True
do_print = False

av_advantage = None
std_advantage = None

grad_norms = None

#train_batch_gen, eval_batch_gen = get_data_generators(
#    train_batch_size, chunk_length, eval_batch_size
#)

with open('data/fasttext/my_de_emb', 'rb') as f:
    de_emb = pickle.load(f)

with open('data/fasttext/my_en_emb', 'rb') as f:
    en_emb = pickle.load(f)

model_params = {
    'vocab_size_encoder': vocab_size_encoder,
    'vocab_size_decoder': vocab_size_decoder,
    'enc_pre_emb': de_emb,
    'dec_pre_emb': en_emb,
    'embed_dim': 128,
    'hidden_size': 256
}

import os
import pickle

save_path = './saved_models/' + mode_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
write_path = './output/' + mode_name
if not os.path.exists(write_path):
    os.makedirs(write_path)

# %% TRAINING
print(mode_name)
for run in range(num_runs):
    print('Run', run)
    print()
    train_losses.append([])
    train_accs.append([])
    cum_train_loss = 0
    cum_train_acc = 0

    train_av_loss = 0
    batch_av_train_av_loss = 0

    eval_losses.append([])
    eval_accs.append([])
    cum_eval_loss = 0
    cum_eval_acc = 0
    cum_eval_bleu = 0

    global_start_time = time()
    last_print_time = global_start_time
    model = CUDA_wrapper(
        Seq2SeqModel(
            **model_params,
            feed_mode=feed_mode, 
            baseline_feed_mode=baseline_feed_mode,
            attention_mode=attention_mode, 
            baseline_attention_mode = baseline_attention_mode
        )
    )

    av_advantage = []
    std_advantage = []

    grad_norms = []
    grad_norms_biased = []

    init_lr = 0.001
    lr = init_lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for step in range(num_steps):
        if softmax_t_range != NOT_AVAILABLE:
            softmax_t = softmax_t_range[0] * (softmax_t_range[1] / softmax_t_range[0]) ** (step / num_steps)
        else:
            softmax_t = 0.0
        tf_ratio = tf_ratio_range[0] + (tf_ratio_range[1] - tf_ratio_range[0]) * (step / num_steps)

        chunk_batch_torch, tgt_batch_torch, tgt_len = bg.next_train()
        #print(chunk_batch_torch.size())
        #print(tgt_batch_torch.size())
        true_train_batch_size, true_chunk_length = chunk_batch_torch.size()
        true_train_batch_size, true_tgt_length = tgt_batch_torch.size()

        (unscaled_logits, outputs), \
        (unscaled_logits_feed_baseline, outputs_feed_baseline), \
        (unscaled_logits_attn_baseline, outputs_attn_baseline) = model(
            chunk_batch_torch, tgt_batch_torch,
            softmax_temperature=softmax_t,
            teacher_forcing_ratio=tf_ratio
        )
        
        if use_masked_loss:
            # unscaled_logits.data.shape
            # rev_chunk_batch.data.shape
            # print('Check (batch, max_len, num_classes) :', unscaled_logits.data.shape)
            # print('Check2 (batch, max_len): ', tgt_batch.data.shape)
            train_loss = loss_function(unscaled_logits, tgt_batch_torch, tgt_len)
        else:
            train_loss = loss_function(unscaled_logits.view(-1, vocab_size_decoder), tgt_batch_torch.view(-1))
        
        train_acc = torch.mean(torch.eq(outputs, tgt_batch_torch).float())
        
        if feed_baseline != NOT_AVAILABLE:
            if feed_baseline == 'no-reinforce':
                for dec_feed in model.decoder.dec_feeds:
                    dec_feed.reinforce(
                        CUDA_wrapper(torch.zeros(true_train_batch_size, 1))
                    )
            elif feed_baseline == 'argmax':
                tgt_batch_torch_one_hot = CUDA_wrapper(
                    torch.zeros(
                        true_train_batch_size, true_tgt_length, vocab_size_decoder
                    )
                )
                tgt_batch_torch_one_hot.scatter_(
                    2, tgt_batch_torch.data.view(true_train_batch_size, true_tgt_length, 1), 1
                )
                elemwise_train_loss = (-1) * F.log_softmax(
                    unscaled_logits.data.view(-1, vocab_size_decoder)
                ).data.view(true_train_batch_size, true_tgt_length, vocab_size_decoder)[tgt_batch_torch_one_hot.byte()].view(
                    true_train_batch_size, true_tgt_length
                )
                elemwise_train_loss_feed_baseline = (-1) * F.log_softmax(
                    unscaled_logits_feed_baseline.data.view(-1, vocab_size_decoder)
                ).data.view(true_train_batch_size, true_tgt_length, vocab_size_decoder)[tgt_batch_torch_one_hot.byte()].view(
                    true_train_batch_size, true_tgt_length
                )
                normed_elemwise_advantage = (
                    (elemwise_train_loss_feed_baseline - elemwise_train_loss) /
                    (true_train_batch_size * true_tgt_length)
                )
                sum_normed_elemwise_advantage = torch.sum(
                    normed_elemwise_advantage, dim=1
                )
                cumsum_normed_elemwise_advantage = torch.cumsum(
                    normed_elemwise_advantage, dim=1
                )
                for t, dec_feed in enumerate(model.decoder.dec_feeds):
                    dec_feed.reinforce(
                        (sum_normed_elemwise_advantage - cumsum_normed_elemwise_advantage[:, t]).view(true_train_batch_size, 1)
                    )

                av_advantage.append(torch.mean(elemwise_train_loss_feed_baseline - elemwise_train_loss, dim=0).cpu().numpy())
                std_advantage.append(torch.std(elemwise_train_loss_feed_baseline - elemwise_train_loss, dim=0).cpu().numpy())
            else:
                raise ValueError('Unknown feed_baseline: {}'.format(feed_baseline))

        if attention_baseline != NOT_AVAILABLE:
            if attention_baseline == 'argmax':
                tgt_batch_torch_one_hot = CUDA_wrapper(
                    torch.zeros(
                        true_train_batch_size, true_tgt_length, vocab_size_decoder
                    )
                )
                tgt_batch_torch_one_hot.scatter_(
                    2, tgt_batch_torch.data.view(true_train_batch_size, true_tgt_length, 1), 1
                )
                elemwise_train_loss = (-1) * F.log_softmax(
                    unscaled_logits.data.view(-1, vocab_size_decoder)
                ).data.view(true_train_batch_size, true_tgt_length, vocab_size_decoder)[tgt_batch_torch_one_hot.byte()].view(
                    true_train_batch_size, true_tgt_length
                )
                elemwise_train_loss_attn_baseline = (-1) * F.log_softmax(
                    unscaled_logits_attn_baseline.data.view(-1, vocab_size_decoder)
                ).data.view(true_train_batch_size, true_tgt_length, vocab_size_decoder)[tgt_batch_torch_one_hot.byte()].view(
                    true_train_batch_size, true_tgt_length
                )
                normed_elemwise_advantage = (
                    (elemwise_train_loss_attn_baseline - elemwise_train_loss) /
                    (true_train_batch_size * true_tgt_length)
                )
                sum_normed_elemwise_advantage = torch.sum(
                    normed_elemwise_advantage, dim=1
                )
                cumsum_normed_elemwise_advantage = torch.cumsum(
                    normed_elemwise_advantage, dim=1
                )
                for t, attn_idx in enumerate(model.decoder.attention_idx):
                    attn_idx.reinforce(
                        (sum_normed_elemwise_advantage - cumsum_normed_elemwise_advantage[:, t]).view(true_train_batch_size, 1)
                    )

            else:
                raise ValueError('Unknown attention_baseline: {}'.format(attention_baseline))

        optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=4)
        optimizer.step()
        train_losses[-1].append(train_loss.data.cpu().numpy().mean())
        train_accs[-1].append(train_acc.data.cpu().numpy().mean())

        cum_train_loss += train_losses[-1][-1]
        cum_train_acc += train_accs[-1][-1]
        if do_eval:
            chunk_batch_torch, tgt_batch_torch, tgt_len = bg.next_eval()

            (unscaled_logits, outputs), \
            (unscaled_logits_feed_baseline, outputs_feed_baseline), \
            (unscaled_logits_attn_baseline, outputs_attn_baseline) = model(
                chunk_batch_torch, tgt_batch_torch,
                work_mode='test'
            )
            if use_masked_loss:
                eval_loss = loss_function(unscaled_logits, tgt_batch_torch, tgt_len)
            else:
                eval_loss = loss_function(
                    unscaled_logits.view(-1, vocab_size_decoder),
                    tgt_batch_torch.view(-1)
                )
            eval_acc = torch.mean(torch.eq(outputs[:, :tgt_batch_torch.size(1)].contiguous(), tgt_batch_torch).float())

            eval_losses[-1].append(eval_loss.data.cpu().numpy().mean())
            eval_accs[-1].append(eval_acc.data.cpu().numpy().mean())

            cum_eval_loss += eval_losses[-1][-1]
            cum_eval_bleu += bleu_score(
                unscaled_logits, outputs, tgt_batch_torch, bg.vocab.tgt.id2word
            )
            cum_eval_acc += eval_accs[-1][-1]

        # Print:
        if (step + 1) % print_skip == 0:
            print('Step', step + 1)

            print('softmax temperature: {:.2f}'.format(softmax_t))
            print('teacher-forcing ratio: {:.2f}'.format(tf_ratio))

            print('Train loss: {:.2f}; train accuracy: {:.2f}'.format(
                cum_train_loss / print_skip, cum_train_acc / print_skip
            ))
            cum_train_loss = 0
            cum_train_acc = 0

            if do_eval:
                if task=='translation':
                    print('Eval loss: {:.2f}; eval accuracy: {:.2f}; eval bleu: {:.2f}'.format(
                        cum_eval_loss / print_skip, cum_eval_acc / print_skip, cum_eval_bleu / print_skip
                    ))
                    cum_eval_loss = 0
                    cum_eval_acc = 0
                    cum_eval_bleu = 0

                    if do_print:
                        outputs_np = outputs.data.cpu().numpy()
                        cur_decode_batch_size = min(
                            decode_batch_size,
                            min(len(tgt_batch_torch), len(outputs_np))
                        )
                        for i in range(cur_decode_batch_size):
                            print('{}\n|  vs  |\n{}'.format(
                                ' '.join([bg.vocab.tgt.id2word[k.data[0]] for k in tgt_batch_torch[i]]),
                                ' '.join([bg.vocab.tgt.id2word[k] for k in outputs_np[i]])
                            ))

                else:
                    raise ValueError('Unknown task: {}'.format(task))

            print('{:.2f}s from last print'.format(time() - last_print_time))
            last_print_time = time()
            print()
        if (step + 1) % save_per_step == 0:
            print('perform saving the model:')
            torch.save(model.state_dict(), save_path + "/step=" + str(step))
            print('model saved')

            eval_losses_mean = np.mean(eval_losses, axis=0)
            eval_losses_std = np.std(eval_losses, axis=0)

            eval_accs_mean = np.mean(eval_accs, axis=0)
            eval_accs_std = np.std(eval_accs, axis=0)

            train_losses_mean = np.mean(train_losses, axis=0)
            train_losses_std = np.std(train_losses, axis=0)

            train_accs_mean = np.mean(train_accs, axis=0)
            train_accs_std = np.std(train_accs, axis=0)

            for name in ['train_losses', 'train_accs', 'eval_losses', 'eval_accs']:
                for suffix in ['', '_mean', '_std']:
                    with open(write_path + '/' + name + suffix + '.dat', 'wb') as f:
                        pickle.dump(eval(name + suffix), f)
            print('output written')
