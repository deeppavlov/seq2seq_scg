import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from gumbel_softmax import gumbel_softmax_sample

import util
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
            print('using pretrained embeddings for encoder with shape:', embeddings.shape)
            self.embed_dim = embeddings.shape[1]
            self.embedding = nn.Embedding(*embeddings.shape)
            self.embedding.weight = nn.Parameter(torch.Tensor(embeddings))
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.embed_dim = embed_dim

        self.enc_rnn = nn.LSTM(self.embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, input_chunk):
        batch_size, seq_length = input_chunk.size()
        
        input_chunk_emb = self.embedding(input_chunk)
        first_hidden_state = autograd.Variable(
            util.Tensor(self.num_layers * 2, batch_size, self.hidden_size).zero_(),
            requires_grad=False
        )
        first_cell_state = autograd.Variable(
            util.Tensor(self.num_layers * 2, batch_size, self.hidden_size).zero_(),
            requires_grad=False
        )

        all_hidden, enc_hc_last = self.enc_rnn(
            input_chunk_emb, (first_hidden_state, first_cell_state)
        ) # ([batch_size x seq_len x self.hidden_size * 2],
          #  ([self.num_layers * 2 x batch_size x self.hidden_size], [self.num_layers * 2 x batch_size x self.hidden_size])
          # )
        return all_hidden, enc_hc_last


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_size, seq_max_len=180,
        num_layers=1, embeddings=None
    ):
        super(Decoder, self).__init__()

        if embeddings is not None:
            print('using pretrained embeddings for decoder with shape: ', embeddings.shape)
            self.embed_dim = embeddings.shape[1]
            self.embedding = nn.Embedding(*embeddings.shape)
            self.embedding.weight = nn.Parameter(torch.Tensor(embeddings))
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_max_len = seq_max_len
        
        self.dec_cell = nn.LSTMCell(hidden_size * 2 + self.embed_dim, hidden_size * num_layers * 2)
        self.output_proj = nn.Linear(hidden_size * num_layers * 2, vocab_size)
        self.W_attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.softmax = nn.Softmax()

    def forward(
        self, all_hidden, last_hc, target_chunk,
        feed_mode, work_mode, output_mode, attention_mode,
        dropout_rate=0.2, softmax_temperature=1.0,
        teacher_forcing_ratio=1.0
    ):
        """
        feed_mode: 'argmax', 'softmax', 'sampling', 'gumbel', 'gumbel-st' or 'same'
        work_mode: 'training' or 'inference'
        output_mode: 'argmax' or 'softmax'
        attention_mode: 'soft', 'hard', 'gumbel' or 'gumbel-st'
        """
        dropout = nn.Dropout(p=dropout_rate)
        
        batch_size, seq_length = target_chunk.size()
        inp_seq_len = all_hidden.size()[1]
        if work_mode != 'training':
            seq_length = self.seq_max_len
        
        dec_unscaled_logits = []
        dec_outputs = []
        self.dec_feeds = []
        self.attention_idx = []

        target_chunk_emb = self.embedding(target_chunk)
        dec_h = torch.cat(last_hc[0], 1) # [batch_size x self.hidden_size * self.num_layers * 2]
        dec_c = torch.cat(last_hc[1], 1)
        
        dec_feed = None
        dec_feed_emb = autograd.Variable(
            util.Tensor(batch_size, self.embed_dim).zero_(),
            requires_grad=False
        )

        ## decoding cycle
        for t in range(seq_length):

            ## calculating attention:
            attention_scores = self.W_attention(all_hidden) #  batch_size x inp_seq_length X hidden_size * 2
            attention_scores = torch.bmm(attention_scores, dec_h.unsqueeze(2)) #  batch_size x inp_seq_length x 1
            attention_scores = self.softmax(
                attention_scores.view(batch_size, inp_seq_len)
            )
            
            if attention_mode == 'soft':
                attention = attention_scores
            elif attention_mode in ['gumbel', 'gumbel-st']:
                attention = gumbel_softmax_sample(
                    attention_scores,
                    # actually, we need to specify temperature here
                    hard=(attention_mode == 'gumbel-st')
                )
            elif attention_mode in ['hard', 'argmax']:
                if attention_mode == 'hard':
                    attention_idx = torch.multinomial(attention_scores, 1)
                elif attention_mode == 'argmax':
                    attention_idx = torch.max(attention_scores, 1)[1]
                attention = autograd.Variable(
                    util.Tensor(batch_size, inp_seq_len).zero_(), requires_grad=False
                )
                attention.scatter_(1, attention_idx.data.view(batch_size, 1), 1)
                self.attention_idx.append(attention_idx)

            context_vector = torch.bmm(attention.view(batch_size, 1, inp_seq_len), all_hidden) # [batch_size x 1 x hidden_size * 2]
            context_vector = context_vector.view(batch_size, self.hidden_size * 2)
            
            concatenated_with_attention_feed = torch.cat(
                (context_vector, dropout(dec_feed_emb)), dim=1
            ) # [batch_size x hidden_size * 2 + embed_size]
            
            dec_h, dec_c = self.dec_cell(concatenated_with_attention_feed, (dec_h, dec_c))
            dec_unscaled_logits.append(self.output_proj(dec_h))
            
            ## generating output:
            # always perform greedy decoding in inference mode:
            if output_mode == 'argmax' or work_mode == 'inference':
                dec_output = torch.max(dec_unscaled_logits[-1], dim=1)[1].view(batch_size, 1)
            elif output_mode == 'sampling':
                dec_output = torch.multinomial(
                    F.softmax(dec_unscaled_logits[-1]), 1
                )
            else:
                raise ValueError("Invalid output_mode: '{}'".format(output_mode))
            # save decoder outputs:
            dec_outputs.append(dec_output.view(batch_size).data)

            ## generating next token to feed in:
            if work_mode == 'inference':
                dec_feed = dec_output
                dec_feed_emb = self.embedding(dec_feed).view(batch_size, -1)
            elif work_mode == 'training':
                if feed_mode in ['argmax', 'sampling']:
                    if feed_mode == 'argmax':
                        dec_feed = torch.max(dec_unscaled_logits[-1], dim=1)[1].view(batch_size, 1)
                    elif feed_mode == 'sampling':
                        dec_feed = torch.multinomial(
                            F.softmax(dec_unscaled_logits[-1]), 1
                        )
                        # save decoder feeds only when feed_mode is 'sampling' or output_mode is 'sampling' and feed_mode is 'same'
                        self.dec_feeds.append(dec_feed)
                    dec_feed_emb = self.embedding(dec_feed).view(batch_size, -1)
                elif feed_mode in ['softmax', 'gumbel', 'gumbel-st']:
                    if feed_mode == 'softmax':
                        dec_feed_distr = F.softmax(
                            dec_unscaled_logits[-1] / softmax_temperature
                        )
                    elif feed_mode in ['gumbel', 'gumbel-st']:
                        dec_feed_distr = gumbel_softmax_sample(
                            F.softmax(dec_unscaled_logits[-1]),
                            softmax_temperature,
                            hard=(feed_mode == 'gumbel_st')
                        )
                    def_feed_emb = torch.matmul(
                        dec_feed_distr, self.embedding.weight
                    )
                elif feed_mode == 'same':
                    dec_feed = dec_output
                    # save decoder feeds only when feed_mode is 'sampling' or output_mode is 'sampling' and feed_mode is 'same'
                    if output_mode == 'sampling':
                        self.dec_feeds.append(dec_output)
                    dec_feed_emb = self.embedding(dec_feed).view(batch_size, -1)
                
                use_teacher_forcing = (random.random() < teacher_forcing_ratio)
                if use_teacher_forcing:
                    dec_feed = target_chunk[:, t]
                    dec_feed_emb = target_chunk_emb[:, t]

        return (
            torch.stack(dec_unscaled_logits, dim=1),
            torch.stack(dec_outputs, dim=1)
        )


class Seq2SeqModel(nn.Module):
    def __init__(
        self, vocab_size_encoder, vocab_size_decoder, embed_dim, hidden_size,
        enc_pre_emb=None, dec_pre_emb=None,
        num_layers_enc=1, num_layers_dec=1,
        output_mode='argmax', baseline_output_mode=None,
        feed_mode='argmax', baseline_feed_mode=None,
        attention_mode='soft', baseline_attention_mode=None
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
                num_layers=num_layers_dec, embeddings=dec_pre_emb
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

        self.output_mode = output_mode
        self.baseline_output_mode = baseline_output_mode
        self.feed_mode = feed_mode
        self.baseline_feed_mode = baseline_feed_mode
        self.attention_mode = attention_mode
        self.baseline_attention_mode = baseline_attention_mode

    def forward(
        self, input_chunk, target_chunk, work_mode='training',
        dropout_rate=0.2, softmax_temperature=1.0,
        teacher_forcing_ratio=1.0
    ):
        all_hidden, last_hc = self.encoder(input_chunk)
        if work_mode == 'inference':
            dropout_rate = 0.0

        need_feed_baseline = self.baseline_feed_mode is not None and work_mode != 'inference'
        need_attn_baseline = self.baseline_attention_mode is not None and work_mode != 'inference'

        if need_feed_baseline:
            feed_baseline_output = self.decoder(
                all_hidden, last_hc, target_chunk,
                output_mode=self.baseline_output_mode,
                feed_mode=self.baseline_feed_mode, work_mode=work_mode,
                attention_mode=self.attention_mode,
                dropout_rate=dropout_rate, softmax_temperature=softmax_temperature,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
        else:
            feed_baseline_output = (None, None)
        if need_attn_baseline:
            attn_baseline_output = self.decoder(
                all_hidden, last_hc, target_chunk,
                output_mode=self.baseline_output_mode,
                feed_mode=self.feed_mode, work_mode=work_mode,
                attention_mode=self.baseline_attention_mode,
                dropout_rate=dropout_rate, softmax_temperature=softmax_temperature,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
        else:
            attn_baseline_output = (None, None)
        output = self.decoder(
            all_hidden, last_hc, target_chunk,
            output_mode=self.output_mode,
            feed_mode=self.feed_mode, work_mode=work_mode,
            attention_mode=self.attention_mode,
            dropout_rate=dropout_rate, softmax_temperature=softmax_temperature,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        return output, feed_baseline_output, attn_baseline_output

