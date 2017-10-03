# from matplotlib import pyplot as plt
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
# plt.rcParams['figure.figsize'] = (15, 9)
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
# %%  <- this symbols mean "cell seporator" in Hydrogen plugin for Atom

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

train_batch_size = 128 # that was 256
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

num_steps = 60000
print_skip = 100
save_per_step = 10000

train_losses = []
train_accs = []

train_losses_pav = []
train_accs_pav = []

eval_losses = []
eval_accs = []

reinforce_strategy = 'none'#'argmax_advantage'

do_eval = True

use_polyak_average = False

av_advantage = None
std_advantage = None

grad_norms = None
grad_norms_biased = None
train_batch_gen, eval_batch_gen = get_data_generators(train_batch_size, chunk_length, eval_batch_size)

# the_model = Seq2SeqModel(vocab_size_encoder=10, vocab_size_decoder=10, embed_dim=8, hidden_size=48)
# %% TRAINING
for run in range(num_runs):
    print('Run', run)
    print()
    train_losses.append([])
    train_accs.append([])
    cum_train_loss = 0
    cum_train_acc = 0

    train_losses_pav.append([])
    train_accs_pav.append([])
    cum_train_loss_pav = 0
    cum_train_acc_pav = 0

    train_av_loss = 0
    batch_av_train_av_loss = 0

    eval_losses.append([])
    eval_accs.append([])
    cum_eval_loss = 0
    cum_eval_acc = 0

    global_start_time = time()
    last_print_time = global_start_time
    model = CUDA_wrapper(Seq2SeqModel(vocab_size_encoder=vocab_size_encoder, vocab_size_decoder=vocab_size_decoder, embed_dim=512, hidden_size=512))

    model_pav = CUDA_wrapper(Seq2SeqModel(vocab_size_encoder=vocab_size_encoder, vocab_size_decoder=vocab_size_decoder, embed_dim=512, hidden_size=512))
    av_advantage = []
    std_advantage = []

    grad_norms = []
    grad_norms_biased = []

    init_lr = 0.0001
    lr = init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(num_steps):
        ######################## words_reverser ########################
        # chunk_batch = next(train_batch_gen)
        # rev_chunk_batch = revert_words(chunk_batch)
        # chunk_batch_torch = autograd.Variable(CUDA_wrapper(torch.from_numpy(chunk_batch)), requires_grad=False)
        # rev_chunk_batch_torch = autograd.Variable(CUDA_wrapper(torch.from_numpy(rev_chunk_batch)), requires_grad=False)
        ######################## end words reverser ########################
        chunk_batch_torch, rev_chunk_batch_torch, tgt_len = bg.next_train()

        if reinforce_strategy == 'argmax_advantage':
            unscaled_logits, unscaled_logits_baseline, outputs = model(
                chunk_batch_torch, rev_chunk_batch_torch,
                # output_mode='argmax', feed_mode='sampling',
                # baseline_mode='argmax'
            )
        else:
            unscaled_logits, outputs = model(
                chunk_batch_torch, rev_chunk_batch_torch,
                # output_mode='argmax', feed_mode='sampling', baseline_mode=None
            )
        if use_masked_loss:
            # unscaled_logits.data.shape
            # rev_chunk_batch.data.shape
            # print('Check (batch, max_len, num_classes) :', unscaled_logits.data.shape)
            # print('Check2 (batch, max_len): ', rev_chunk_batch.data.shape)
            train_loss = loss_function(unscaled_logits, rev_chunk_batch_torch, tgt_len)
        else:
            train_loss = loss_function(unscaled_logits.view(-1, vocab_size_decoder), rev_chunk_batch_torch.view(-1))
        train_acc = torch.mean(torch.eq(outputs, rev_chunk_batch_torch).float())
        if reinforce_strategy == 'none':
            pass
            # TODO: ask Zhenya why reinforce zeros on not stochastic nodes ????
            # for t, dec_feed in enumerate(model.decoder.dec_feeds):
            #     dec_feed.reinforce(CUDA_wrapper(torch.zeros(train_batch_size, 1)))
        elif reinforce_strategy == 'argmax_advantage':
            rev_chunk_batch_torch_one_hot = CUDA_wrapper(torch.zeros(train_batch_size, chunk_length, vocab_size))
            rev_chunk_batch_torch_one_hot.scatter_(
                2, rev_chunk_batch_torch.data.view(train_batch_size, chunk_length, 1), 1
            )
            elemwise_train_loss = (-1) * F.log_softmax(
                unscaled_logits.data.view(-1, vocab_size)
            ).data.view(train_batch_size, chunk_length, vocab_size)[rev_chunk_batch_torch_one_hot.byte()].view(
                train_batch_size, chunk_length
            )
            elemwise_train_loss_baseline = (-1) * F.log_softmax(
                unscaled_logits_baseline.data.view(-1, vocab_size)
            ).data.view(train_batch_size, chunk_length, vocab_size)[rev_chunk_batch_torch_one_hot.byte()].view(
                train_batch_size, chunk_length
            )
            normed_elemwise_advantage = ((elemwise_train_loss_baseline - elemwise_train_loss) /
                                         (train_batch_size * chunk_length))
            sum_normed_elemwise_advantage = torch.sum(normed_elemwise_advantage, dim=1)
            cumsum_normed_elemwise_advantage = torch.cumsum(normed_elemwise_advantage, dim=1)
            for t, dec_feed in enumerate(model.decoder.dec_feeds):
                dec_feed.reinforce(
                    (sum_normed_elemwise_advantage - cumsum_normed_elemwise_advantage[:, t]).view(train_batch_size, 1)
                )

            av_advantage.append(torch.mean(elemwise_train_loss_baseline - elemwise_train_loss, dim=0).cpu().numpy())
            std_advantage.append(torch.std(elemwise_train_loss_baseline - elemwise_train_loss, dim=0).cpu().numpy())

        optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=4)
        optimizer.step()
        train_losses[run].append(train_loss.data.cpu().numpy().mean())
        train_accs[run].append(train_acc.data.cpu().numpy().mean())

        cum_train_loss += train_losses[run][-1]
        cum_train_acc += train_accs[run][-1]
        if do_eval:
            ######################## words_reverser ########################
            # chunk_batch = next(eval_batch_gen)
            # rev_chunk_batch = revert_words(chunk_batch)
            #
            # chunk_batch_torch = autograd.Variable(CUDA_wrapper(torch.from_numpy(chunk_batch)), requires_grad=False)
            # rev_chunk_batch_torch = autograd.Variable(CUDA_wrapper(torch.from_numpy(rev_chunk_batch)), requires_grad=False)
            ######################## end words_reverser ########################

            chunk_batch_torch, rev_chunk_batch_torch, tgt_len = bg.next_eval()

            unscaled_logits, outputs = model(
                chunk_batch_torch, rev_chunk_batch_torch,
                # output_mode='argmax', feed_mode='sampling'
            )
            if use_masked_loss:
                eval_loss = loss_function(unscaled_logits, rev_chunk_batch_torch, tgt_len)
            else:
                eval_loss = loss_function(unscaled_logits.view(-1, vocab_size_decoder), rev_chunk_batch_torch.view(-1))
            eval_acc = torch.mean(torch.eq(outputs, rev_chunk_batch_torch).float())

            eval_losses[run].append(eval_loss.data.cpu().numpy().mean())
            eval_accs[run].append(eval_acc.data.cpu().numpy().mean())

            cum_eval_loss += eval_losses[run][-1]
            cum_eval_acc += eval_accs[run][-1]

        # Print:
        if (step + 1) % print_skip == 0:
            print('Step', step + 1)

            print('Train loss: {:.2f}; train accuracy: {:.2f}'.format(
                cum_train_loss / print_skip, cum_train_acc / print_skip
            ))
            cum_train_loss = 0
            cum_train_acc = 0

            if use_polyak_average:
                print('Train loss for polyak-averaged model: {:.2f}; accuracy: {:.2f}'.format(
                    cum_train_loss_pav / print_skip, cum_train_acc_pav / print_skip
                ))
                cum_train_loss_pav = 0
                cum_train_acc_pav = 0

            if do_eval:
                if task=='translation':
                    print('Eval loss: {:.2f}; eval accuracy: {:.2f}'.format(
                        cum_eval_loss / print_skip, cum_eval_acc / print_skip
                    ))
                    cum_eval_loss = 0
                    cum_eval_acc = 0

                    outputs_np = outputs.data.cpu().numpy()
                    cur_decode_batch_size = min(decode_batch_size, min(len(rev_chunk_batch_torch), len(outputs_np)))
                    for i in range(cur_decode_batch_size):
                        print('{}|  vs  |{}'.format(
                            ' '.join([bg.vocab.tgt.id2word[k.data[0]] for k in rev_chunk_batch_torch[i]]),
                            ' '.join([bg.vocab.tgt.id2word[k.data[0]] for k in outputs_np[i]])
                        ))
                else:
                    print('Eval loss: {:.2f}; eval accuracy: {:.2f}'.format(
                        cum_eval_loss / print_skip, cum_eval_acc / print_skip
                    ))
                    cum_eval_loss = 0
                    cum_eval_acc = 0

                    outputs_np = outputs.data.cpu().numpy()
                    for i in range(decode_batch_size):
                        print('{}|  vs  |{}'.format(
                            ''.join(list(map(id_to_char, chunk_batch[i]))),
                            ''.join(list(map(id_to_char, outputs_np[i])))
                        ))

            print('{:.2f}s from last print'.format(time() - last_print_time))
            last_print_time = time()
            print()
        if (step + 1) % save_per_step == 0:
            print('perform saving the model:')
            torch.save(model.state_dict(), "./saved_model_step=" + str(step)  + "_time=" + str(datetime.datetime.now()))
            print('model saved')
