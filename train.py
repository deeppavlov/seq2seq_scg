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
from utils.misc import CUDA_wrapper, masked_cross_entropy, bleu_score

import os
import pickle
from time import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--do_eval', action='store_true', default=False)
parser.add_argument('--do_print', action='store_true', default=False)

args = parser.parse_args()

do_eval = args.do_eval
do_print = args.do_print

train_params = param_provider.get_train_params()
model_params = param_provider.get_model_params()

for key, val in train_params.items():
    exec(key + '=val')
for key, val in model_params.items():
    exec(key + '=val')

batch_generator = BatchGenerator(
    vocab_path=vocab_path, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, test_batch_size=test_batch_size
)

if use_masked_loss:
    loss_function = masked_cross_entropy
else:
    loss_function = nn.CrossEntropyLoss()

vocab_size_encoder = len(batch_generator.vocab.src)
vocab_size_decoder = len(batch_generator.vocab.tgt)

mode_name = 'opt_goal=' + opt_goal + \
            '__output=' + output_mode + '__output_baseline=' + output_baseline + \
            '__feed=' + feed_mode + '__feed_baseline=' + feed_baseline + \
            '__attn=' + attention_mode + '__attn_baseline=' + attention_baseline +\
            '__tf_ratio=' + str(tf_ratio_range) + \
            '__softmax_t=' + str(softmax_t_range) + \
            '__init_lr=' + str(init_lr)

with open('data/fasttext/my_de_emb', 'rb') as f:
    de_emb = pickle.load(f)

with open('data/fasttext/my_en_emb', 'rb') as f:
    en_emb = pickle.load(f)

technical_params = {
    'vocab_size_encoder': vocab_size_encoder,
    'vocab_size_decoder': vocab_size_decoder,
    'enc_pre_emb': de_emb if start_model is None else None,
    'dec_pre_emb': en_emb if start_model is None else None,
    'embed_dim': 300,
    'hidden_size': 256
}

save_path = './saved_models/' + mode_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
write_path = './model_outputs/' + mode_name
if not os.path.exists(write_path):
    os.makedirs(write_path)

train_losses = []
train_bleus = []

eval_losses = []
eval_bleus = []

# %% TRAINING
print('training model:', mode_name)
for run in range(num_runs):
    print('Run', run)
    print()
    train_losses.append([])
    train_bleus.append([])
    cum_train_loss = 0
    cum_train_bleu = 0

    eval_losses.append([])
    eval_bleus.append([])
    cum_eval_loss = 0
    cum_eval_bleu = 0

    global_start_time = time()
    last_print_time = global_start_time
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
    if start_model:
        print('starting from model:', start_model)
        model.load_state_dict(torch.load(os.path.join('./saved_models', start_model)))
    lr = init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(start_step, start_step + num_steps, 1):
        if softmax_t_range != param_provider.NOT_AVAILABLE:
            softmax_t = softmax_t_range[0] * (softmax_t_range[1] / softmax_t_range[0]) ** (step / num_steps)
        else:
            softmax_t = 0.0
        tf_ratio = tf_ratio_range[0] + (tf_ratio_range[1] - tf_ratio_range[0]) * (step / num_steps)

        chunk_batch_torch, tgt_batch_torch, tgt_len = batch_generator.next_train()
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
            train_loss = loss_function(unscaled_logits, tgt_batch_torch, tgt_len)
        else:
            train_loss = loss_function(unscaled_logits.view(-1, vocab_size_decoder), tgt_batch_torch.view(-1))
            
        train_bleu = bleu_score(outputs, tgt_batch_torch, batch_generator.vocab.tgt.id2word)
        
        if feed_baseline != param_provider.NOT_AVAILABLE:
            if feed_baseline == 'no-reinforce':
                for dec_feed in model.decoder.dec_feeds:
                    dec_feed.reinforce(
                        util.Tensor(true_train_batch_size, 1).zero_()
                    )
            elif feed_baseline == 'argmax':
                tgt_batch_torch_one_hot = util.Tensor(
                    true_train_batch_size, true_tgt_length, vocab_size_decoder
                ).zero_()
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

            elif feed_baseline == 'av_loss':
                av_loss_alpha = 0.99
                if step == 0:
                    train_av_loss = 0
                tgt_batch_torch_one_hot = util.Tensor(
                    true_train_batch_size, true_tgt_length, vocab_size_decoder
                ).zero_()
                tgt_batch_torch_one_hot.scatter_(
                    2, tgt_batch_torch.data.view(true_train_batch_size, true_tgt_length, 1), 1
                )
                elemwise_train_loss = (-1) * F.log_softmax(
                    unscaled_logits.data.view(-1, vocab_size_decoder)
                ).data.view(true_train_batch_size, true_tgt_length, vocab_size_decoder)[tgt_batch_torch_one_hot.byte()].view(
                    true_train_batch_size, true_tgt_length
                )
                normed_elemwise_advantage = (-1) * (elemwise_train_loss - train_av_loss) / (true_train_batch_size * true_tgt_length)
                for t, dec_feed in enumerate(model.decoder.dec_feeds):
                    dec_feed.reinforce(
                        normed_elemwise_advantage[:, t].contiguous().view(
                            true_train_batch_size, 1
                        )
                    )
                train_av_loss = av_loss_alpha * train_av_loss + \
                        (1 - av_loss_alpha) * train_loss.data

            else:
                raise ValueError('Unknown feed_baseline: {}'.format(feed_baseline))

        if attention_baseline != param_provider.NOT_AVAILABLE:
            if attention_baseline == 'argmax':
                tgt_batch_torch_one_hot = util.Tensor(
                    true_train_batch_size, true_tgt_length, vocab_size_decoder
                ).zero_()
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
        if opt_goal == 'log-likelihood':
            train_loss.backward()
        elif opt_goal == 'BLEU':
            train_bleu_sentencewise = bleu_score(
                outputs, tgt_batch_torch, batch_generator.vocab.tgt.id2word,
                corpus_average=False
            )
            if output_baseline == 'av_loss':
                av_bleu_alpha = 0.999
                if step == 0:
                    train_av_bleu = train_bleu
                bleu_advantage = np.array(train_bleu_sentencewise) - train_av_bleu
            elif output_baseline == 'argmax':
                train_bleu_sentencewise_baseline = bleu_score(
                    outputs_feed_baseline, tgt_batch_torch, batch_generator.vocab.tgt.id2word,
                    corpus_average=False
                )
                bleu_advantage = np.array(train_bleu_sentencewise) - \
                                 np.array(train_bleu_sentencewise_baseline)
            else:
                raise ValueError(
                    "Unknown output_baseline: '{}'".format(output_baseline)
                )
            for dec_feed in model.decoder.dec_feeds:
                dec_feed.reinforce(
                    CUDA_wrapper(torch.from_numpy(bleu_advantage).float()).view(
                        true_train_batch_size, 1
                    )
                )
            autograd.backward(
                model.decoder.dec_feeds, [None for _ in model.decoder.dec_feeds]
            )
            if output_baseline == 'av_loss':
                train_av_bleu = av_bleu_alpha * train_av_bleu +\
                                (1 - av_bleu_alpha) * np.mean(train_bleu_sentencewise)
        else:
            raise ValueError("Unknown opt_goal: '{}'".format(opt_goal))
        nn.utils.clip_grad_norm(model.parameters(), max_norm=4)
        optimizer.step()

        train_losses[-1].append(train_loss.data.cpu().numpy().mean())
        train_bleus[-1].append(train_bleu)

        cum_train_loss += train_losses[-1][-1]
        cum_train_bleu += train_bleus[-1][-1]
        if do_eval:
            chunk_batch_torch, tgt_batch_torch, tgt_len = batch_generator.next_eval()

            (unscaled_logits, outputs), \
            (unscaled_logits_feed_baseline, outputs_feed_baseline), \
            (unscaled_logits_attn_baseline, outputs_attn_baseline) = model(
                chunk_batch_torch, tgt_batch_torch,
                work_mode='inference'
            )
            if use_masked_loss:
                eval_loss = loss_function(unscaled_logits, tgt_batch_torch, tgt_len)
            else:
                eval_loss = loss_function(
                    unscaled_logits.view(-1, vocab_size_decoder),
                    tgt_batch_torch.view(-1)
                )
            
            eval_bleu = bleu_score(outputs, tgt_batch_torch, batch_generator.vocab.tgt.id2word)

            eval_losses[-1].append(eval_loss.data.cpu().numpy().mean())
            eval_bleus[-1].append(eval_bleu)

            cum_eval_loss += eval_losses[-1][-1]
            cum_eval_bleu += eval_bleus[-1][-1]

        # Print:
        if (step + 1) % print_skip == 0:
            print('Step', step + 1)

            print('softmax temperature: {:.2f}'.format(softmax_t))
            print('teacher-forcing ratio: {:.2f}'.format(tf_ratio))

            print('Train loss: {:.2f}; train BLEU: {:.2f}'.format(
                cum_train_loss / print_skip, cum_train_bleu / print_skip
            ))
            cum_train_loss = 0
            cum_train_bleu = 0

            if do_eval:
                print('Eval loss: {:.2f}; eval BLEU: {:.2f}'.format(
                    cum_eval_loss / print_skip, cum_eval_bleu / print_skip
                ))
                cum_eval_loss = 0
                cum_eval_bleu = 0

                if do_print:
                    outputs_np = outputs.cpu().numpy()
                    cur_decode_batch_size = min(
                        decode_batch_size,
                        min(len(tgt_batch_torch), len(outputs_np))
                    )
                    for i in range(cur_decode_batch_size):
                        print('{}\n|  vs  |\n{}'.format(
                            ' '.join([batch_generator.vocab.tgt.id2word[k.data[0]] for k in tgt_batch_torch[i]]),
                            ' '.join([batch_generator.vocab.tgt.id2word[k] for k in outputs_np[i]])
                        ))

            print('{:.2f}s from last print'.format(time() - last_print_time))
            last_print_time = time()
            print()
        if (step + 1) % save_per_step == 0:
            print('perform saving the model:')
            torch.save(model.state_dict(), save_path + "/step=" + str(step))
            print('model saved')

            for name in ['train_losses', 'train_bleus', 'eval_losses', 'eval_bleus']:
                with open(write_path + '/' + name + '.dat', 'wb') as f:
                    pickle.dump(eval(name), f)
            print('output written')
