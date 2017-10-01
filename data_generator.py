# %load_ext autoreload
# %autoreload 2

import numpy as np
import torch
from vocab import Vocab, VocabEntry
from util import read_corpus, batch_slice, data_iter, to_input_variable

class BatchGenerator:
    def train_data_generator(self, batch_size=64):
        self.train_src = "data/nmt_iwslt/train.de-en.de.wmixerprep"
        self.train_tgt = "data/nmt_iwslt/train.de-en.en.wmixerprep"
        train_data_src = read_corpus(self.train_src, source='src')
        train_data_tgt = read_corpus(self.train_tgt, source='tgt')
        return data_iter(zip(train_data_src, train_data_tgt), batch_size=batch_size)

    def eval_data_generator(self, batch_size=64):
        self.dev_src = "data/nmt_iwslt/valid.de-en.de"
        self.dev_tgt = "data/nmt_iwslt/valid.de-en.en"
        eval_data_src = read_corpus(self.dev_src, source='src')
        eval_data_tgt = read_corpus(self.dev_tgt, source='tgt')
        return data_iter(zip(eval_data_src, eval_data_tgt), batch_size=batch_size)

    def data_generator(self, train_batch_size=64, eval_batch_size=64):
        return self.train_data_generator(train_batch_size), self.eval_data_generator(eval_batch_size)

    def __init__(self, vocab_path, train_batch_size=64, eval_batch_size=64):
        self.vocab = torch.load(vocab_path)
        self.data_gen_train, self.data_gen_eval = self.data_generator(train_batch_size, eval_batch_size)
        self.train_batch_size=train_batch_size
        self.eval_batch_size=eval_batch_size

    def next_train(self):
        try:
            src_sents, tgt_sents = next(self.data_gen_train)
        except StopIteration:
            self.data_gen_train = self.train_data_generator(batch_size=self.train_batch_size)
            src_sents, tgt_sents = next(self.data_gen_train)
        inp_src = to_input_variable(src_sents, vocab=self.vocab.src)
        inp_tgt = to_input_variable(tgt_sents, vocab=self.vocab.tgt)
        return torch.transpose(inp_src, 0, 1).contiguous(), torch.transpose(inp_tgt, 0, 1).contiguous()

    def next_eval(self):
        try:
            src_sents, tgt_sents = next(self.data_gen_eval)
        except StopIteration:
            self.data_gen_eval = self.eval_data_generator(batch_size=self.eval_batch_size)
            src_sents, tgt_sents = next(self.data_gen_eval)
        inp_src = to_input_variable(src_sents, vocab=self.vocab.src)
        inp_tgt = to_input_variable(tgt_sents, vocab=self.vocab.tgt)
        return torch.transpose(inp_src, 0, 1).contiguous(), torch.transpose(inp_tgt, 0, 1).contiguous()
