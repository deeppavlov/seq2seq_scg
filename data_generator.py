# %load_ext autoreload
# %autoreload 2

import numpy as np
import torch
from vocab import Vocab, VocabEntry
from util import read_corpus, batch_slice, data_iter, to_input_variable


def data_generator(batch_size=64):
    train_src="data/nmt_iwslt/train.de-en.de.wmixerprep"
    train_tgt="data/nmt_iwslt/train.de-en.en.wmixerprep"
    dev_src="data/nmt_iwslt/valid.de-en.de"
    dev_tgt="data/nmt_iwslt/valid.de-en.en"
    test_src="data/nmt_iwslt/test.de-en.de"
    test_tgt="data/nmt_iwslt/test.de-en.en"
    vocab="data/nmt_iwslt/vocab.bin"

    train_data_src = read_corpus(train_src, source='src')
    train_data_tgt = read_corpus(train_tgt, source = 'tft')


    dev_data_src = read_corpus(dev_src, source='src')
    dev_data_tgt = read_corpus(dev_tgt, source='tgt')

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)

    return data_iter(train_data, batch_size=batch_size)

class BatchGenerator:
    def __init__(self, vocab_path):
        self.vocab = torch.load(vocab_path)
        self.data_gen = data_generator()

    def __iter__(self):
        return self

    def next(self):
        src_sents, tgt_sents = next(self.data_gen)
        inp_src = to_input_variable(src_sents, vocab=self.vocab.src)
        inp_tgt = to_input_variable(tgt_sents, vocab=self.vocab.tgt)
        return torch.transpose(inp_src, 0, 1).contiguous(), torch.transpose(inp_tgt, 0, 1).contiguous()
