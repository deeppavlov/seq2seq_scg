import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import LongTensor
import torch.autograd as autograd
from util import to_input_variable
from util import CUDA_wrapper
import random


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, embeddings=None):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if embeddings is not None:
            print('using pretrained embeddings for encoder with shape: ', embeddings.shape)
            self.embed_dim = embeddings.shape[1]
            self.embedding = nn.Embedding(*embeddings.shape)
            self.embedding.weight = nn.Parameter(torch.FloatTensor(embeddings))
            # self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.embed_dim = embed_dim
        self.enc_rnn = nn.LSTM(self.embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, input_chunk):
        batch_size, seq_length = input_chunk.size()
        self.batch_size = batch_size
        input_chunk_emb = self.embedding(input_chunk)
        first_hidden_state = autograd.Variable(
            CUDA_wrapper(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)),
            requires_grad=False
        )
        first_cell_state = autograd.Variable(
            CUDA_wrapper(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)),
            requires_grad=False
        )
        all_hidden, enc_hc_last = self.enc_rnn(input_chunk_emb, (first_hidden_state, first_cell_state))
        return all_hidden, enc_hc_last

class Decoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_size, seq_max_len=60,
        num_layers=1, embeddings=None, dropout_rate=0.2,
        teacher_forcing_ratio=1., softmax_temperature=1.
    ):
        super(Decoder, self).__init__()
        if embeddings is not None:
            print('using pretrained embeddings for decoder with shape: ', embeddings.shape)
            self.embed_dim = embeddings.shape[1]
            self.embedding = nn.Embedding(*embeddings.shape)
            self.embedding.weight = nn.Parameter(torch.FloatTensor(embeddings))
            # self.embedding.weight.requires_grad = False
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dec_cell = nn.LSTMCell(hidden_size * 2 + self.embed_dim, hidden_size * num_layers * 2)
        self.output_proj = nn.Linear(hidden_size * num_layers * 2, vocab_size)
        self.W_attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax()
        self.seq_max_len = seq_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.softmax_temperature = softmax_temperature
        # self.work_mode = work_mode

    def forward(self, all_hidden, last_hc, target_chunk, batch_size, feed_mode, work_mode='training', output_mode='argmax'):
        dec_feed = None
        batch_size, seq_length = target_chunk.size()
        inp_seq_len = all_hidden.size()[1]
        dec_feed_emb = autograd.Variable(
            CUDA_wrapper(torch.zeros(batch_size, self.embed_dim)),
            requires_grad=False
        )
        dec_unscaled_logits = []
        dec_outputs = []
        self.dec_feeds = []

        target_chunk_emb = self.embedding(target_chunk)
        dec_h = torch.cat(last_hc[0], 1)# all_hidden[:, -1, :] # last hidden:[batch_size x hidden_size * 2]
        dec_c = torch.cat(last_hc[1], 1)#all_cell[:, -1, :]
        if work_mode != 'training':
            seq_length = self.seq_max_len
        self.attention = []
        for t in range(seq_length):
            ## calculate attention
            attention = self.W_attention(all_hidden) #  batch_size x inp_seq_length X hidden_size * 2
            attention = torch.bmm(attention, dec_h.unsqueeze(2)) #  batch_size x inp_seq_length x 1
            attention = self.softmax(attention.view(batch_size, inp_seq_len))
            self.attention.append(attention)
            context_vector = torch.bmm(attention.view(batch_size, 1, inp_seq_len), all_hidden) # [batch_size x 1 x hidden_size * 2]
            context_vector = context_vector.view(batch_size, self.hidden_size * 2)
            concatenated_with_attention_feed = torch.cat((context_vector, self.dropout(dec_feed_emb)), dim=1) # [batch_size x hidden_size * 2 + embed_size]
            dec_h, dec_c = self.dec_cell(concatenated_with_attention_feed, (dec_h, dec_c))
            dec_unscaled_logits.append(self.output_proj(dec_h))
            if output_mode == 'argmax':
                wid = torch.max(dec_unscaled_logits[-1], dim=1)[1]
                dec_outputs.append(wid)
            elif output_mode == 'sampling':
                wid = torch.multinomial(
                    torch.exp(dec_unscaled_logits[-1]), 1
                ).view(batch_size)
                dec_outputs.append(wid)
            else:
                raise ValueError("Invalid output_mode: '{}'".format(output_mode))

            if work_mode == 'test':
                dec_feed = dec_outputs[-1]
                dec_feed_emb = self.embedding(
                    dec_feed.view(batch_size, 1)
                ).view(batch_size, -1)
                self.dec_feeds.append(dec_feed)
            elif work_mode == 'training':
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
                if use_teacher_forcing:
                    dec_feed = target_chunk[:, t]
                    dec_feed_emb = target_chunk_emb[:, t]
                    self.dec_feeds.append(dec_feed)
                else:
                    if feed_mode in ['argmax', 'sampling']:
                        if feed_mode == 'argmax':
                            dec_feed = torch.max(dec_unscaled_logits[-1], dim=1)[1]
                        elif feed_mode == 'sampling':
                            dec_feed = torch.multinomial(
                                F.softmax(dec_unscaled_logits[-1]), 1
                            )
                        dec_feed_emb = self.embedding(
                            dec_feed.view(batch_size, 1)
                        ).view(batch_size, -1)
                        self.dec_feeds.append(dec_feed)
                    elif feed_mode in ['softmax', 'gumbel', 'gumbel-st']:
                        if feed_mode == 'softmax':
                            dec_feed_distr = F.softmax(
                                dec_unscaled_logits[-1] / self.softmax_temperature
                            )
                        elif feed_mode in ['gumbel', 'gumbel-st']:
                            dec_feed_distr = gumbel_softmax_sample(
                                F.softmax(dec_unscaled_logits[-1]),
                                self.softmax_temperature,
                                hard=(feed_mode == 'gumbel_st')
                            )
                        def_feed_emb = torch.matmul(
                            dec_feed_distr, self.embedding.weight
                        )
        return (
            torch.stack(dec_unscaled_logits, dim=1),
            torch.stack(dec_outputs, dim=1)
        )

class Seq2SeqModel(nn.Module):
    def __init__(
        self, vocab_size_encoder, vocab_size_decoder, embed_dim, hidden_size,
        enc_pre_emb=None, dec_pre_emb=None,
        num_layers_enc=1, num_layers_dec=1, dropout_rate=0.2,
        feed_mode='argmax', baseline_feed_mode=None,
        teacher_forcing_ratio=1.0
    ):
        super(Seq2SeqModel, self).__init__()
        self.encoder = CUDA_wrapper(
            Encoder(
                vocab_size_encoder, embed_dim, hidden_size,
                num_layers=num_layers_enc, embeddings=enc_pre_emb
            )
        )
        self.decoder = CUDA_wrapper(
            Decoder(
                vocab_size_decoder, embed_dim, hidden_size,
                num_layers=num_layers_dec, embeddings=dec_pre_emb,
                dropout_rate=dropout_rate,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
        )
        self.vocab_size_encoder = vocab_size_encoder
        self.vocab_size_decoder = vocab_size_decoder
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.output_proj.bias.data.fill_(0)

        self.feed_mode = feed_mode
        self.baseline_feed_mode = baseline_feed_mode
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, input_chunk, target_chunk, work_mode='training'):
        all_hidden, last_hc = self.encoder(input_chunk)
        if self.baseline_feed_mode is None or work_mode == 'test':
            return self.decoder(
                all_hidden, last_hc, target_chunk, self.encoder.batch_size,
                feed_mode=self.feed_mode, work_mode=work_mode,
                teacher_forcing_ratio=self.teacher_forcing_ratio
            )
        else:
            return self.decoder(
                all_hidden, last_hc, target_chunk, self.encoder.batch_size,
                feed_mode=self.feed_mode, work_mode=work_mode,
                teacher_forcing_ratio=self.teacher_forcing_ratio
            ), self.decoder(
                all_hidden, last_hc, target_chunk, self.encoder.batch_size,
                feed_mode=self.baseline_feed_mode, work_mode=work_mode,
                teacher_forcing_ratio=self.teacher_forcing_ratio
            )
