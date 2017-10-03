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


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, n_layers=1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.enc_rnn_cell = nn.LSTMCell(embed_dim, hidden_size)
        self.enc_rnn_cell_dir2 = nn.LSTMCell(embed_dim, hidden_size)
        self.enc_rnn = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

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
        # hidden_enc_all_dir1 = []
        # cell_enc_all_dir1 = []
        # hidden_enc_all_dir2 = []
        # cell_enc_all_dir2 = []
        # hi = autograd.Variable(
        #     CUDA_wrapper(torch.zeros(self.batch_size, self.hidden_size))
        # )
        # ci = autograd.Variable(
        #     CUDA_wrapper(torch.zeros(self.batch_size, self.hidden_size))
        # )
        # for word_i in input_chunk_emb.chunk(input_chunk_emb.size(1), dim=1):
        #     hi, ci = self.enc_rnn_cell(word_i.contiguous().view(batch_size, self.embed_dim), (hi, ci))
        #     hidden_enc_all_dir1.append(hi)
        #     cell_enc_all_dir1.append(ci)
        # for word_i in reversed(input_chunk_emb.chunk(input_chunk_emb.size(1), dim=1)):
        #     hi, ci = self.enc_rnn_cell_dir2(word_i.contiguous().view(batch_size, self.embed_dim), (hi, ci))
        #     hidden_enc_all_dir2.append(hi)
        #     cell_enc_all_dir2.append(ci)
        # # stack hidden and cell lists
        # hidden_dir1 = torch.stack(hidden_enc_all_dir1, dim = 0)
        # cell_dir1 = torch.stack(cell_enc_all_dir1, dim = 0)
        # hidden_dir2 = torch.stack(hidden_enc_all_dir2, dim = 0)
        # cell_dir2 = torch.stack(cell_enc_all_dir2, dim = 0)
        # # cat all directions of hidden and cell
        # all_hidden = torch.cat((hidden_dir1, hidden_dir2), dim = 2)
        # all_hidden = torch.transpose(all_hidden, 0, 1).contiguous()
        # all_cell = torch.cat((cell_dir1, cell_dir2), dim = 2)
        # all_cell = torch.transpose(all_cell, 0, 1).contiguous()
        # return all_hidden, all_cell
        return all_hidden, enc_hc_last

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding, embed_dim, hidden_size, work_mode='training', seq_max_len=20, num_layers=1, dropout_rate=0.2):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dec_cell = nn.LSTMCell(hidden_size * 2 + embed_dim, hidden_size * num_layers * 2)
        self.output_proj = nn.Linear(hidden_size * num_layers * 2, vocab_size)
        self.W_attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax()
        self.seq_max_len = seq_max_len
        self.work_mode = work_mode

    def forward(self, all_hidden, last_hc, target_chunk, batch_size, feed_mode, output_mode='argmax'):
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

        target_chunk_emb = None
        target_chunk_emb = self.embedding(target_chunk)
        dec_h = torch.cat(last_hc[0], 1)# all_hidden[:, -1, :] # last hidden:[batch_size x hidden_size * 2]
        dec_c = torch.cat(last_hc[1], 1)#all_cell[:, -1, :]
        if self.work_mode == 'test':
            seq_length = self.seq_max_len
        for t in range(seq_length):
            ## calculate attention
            attention = self.W_attention(all_hidden) #  batch_size x inp_seq_length X hidden_size * 2
            attention = torch.bmm(attention, dec_h.unsqueeze(2)) #  batch_size x inp_seq_length x 1
            attention = self.softmax(attention.view(batch_size, inp_seq_len))
            context_vector = torch.bmm(attention.view(batch_size, 1, inp_seq_len), all_hidden) # [batch_size x 1 x hidden_size * 2]
            context_vector = context_vector.view(batch_size, self.hidden_size * 2)
            concatenated_with_attention_feed = torch.cat((context_vector, dec_feed_emb), dim=1) # [batch_size x hidden_size * 2 + embed_size]
            dec_h, dec_c = self.dec_cell(self.dropout(concatenated_with_attention_feed), (dec_h, dec_c))
            dec_unscaled_logits.append(self.output_proj(dec_h))
            if output_mode == 'argmax':
                wid = torch.max(dec_unscaled_logits[-1], dim=1)[1]
                dec_outputs.append(wid)
            elif output_mode == 'sampling':
                dec_outputs.append(torch.multinomial(torch.exp(dec_unscaled_logits[-1]), 1).view(batch_size))
            else:
                raise ValueError("Invalid output_mode: '{}'".format(output_mode))

            # feedmode teacher_forcing
            if feed_mode == "teacher_forcing":
                dec_feed = target_chunk[:, t]
                dec_feed_emb = target_chunk_emb[:, t]
                self.dec_feeds.append(dec_feed)
        return (
            torch.stack(dec_unscaled_logits, dim=1),
            torch.stack(dec_outputs, dim=1)
        )

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size_encoder, vocab_size_decoder, embed_dim, hidden_size, work_mode='training', num_layers=1, dropout_rate=0.2):
        super(Seq2SeqModel, self).__init__()
        self.encoder = CUDA_wrapper(Encoder(vocab_size_encoder, embed_dim, hidden_size))
        self.decoder = CUDA_wrapper(Decoder(vocab_size_decoder, self.encoder.embedding, embed_dim, hidden_size, work_mode=work_mode, num_layers=1, dropout_rate=dropout_rate))
        self.vocab_size_encoder = vocab_size_encoder
        self.vocab_size_decoder = vocab_size_decoder
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

    def forward(self, input_chunk, target_chunk, feed_mode="teacher_forcing", num_layers=1, n_layers=1):
        all_hidden, last_hc = self.encoder(input_chunk)
        return self.decoder(all_hidden, last_hc, target_chunk, self.encoder.batch_size, feed_mode)
