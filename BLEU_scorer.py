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


model = Seq2SeqModel(vocab_size_encoder=vocab_size_encoder, vocab_size_decoder=vocab_size_decoder, embed_dim=512, hidden_size=1024)
model.load_state_dict(torch.load('saved_model_step=19999_time=2017-10-03 18:00:33.059905'))

# %%
bleu_scores = []
for _ in range(100):
    chunk_batch_torch, rev_chunk_batch_torch, tgt_len = bg.next_eval()
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
    rev_chunk_batch_torch.data.shape
    unscaled_logits, outputs = model(chunk_batch_torch, rev_chunk_batch_torch, work_mode='test')
    outputs.data.shape
    rev_chunk_batch_torch.data.shape
    hypothesis = transform_tensor_to_list_of_snts(outputs, bg.vocab.tgt.id2word)
    reference = transform_tensor_to_list_of_snts(rev_chunk_batch_torch, bg.vocab.tgt.id2word)
    reference = [[cur_ref] for cur_ref in reference]
    list_of_hypotheses = hypothesis
    list_of_references = reference
    bleu_scores.append(nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses))

np.mean(bleu_scores)

# %% NOT BLEU ###############
# %%
print('value')
embedding = nn.Embedding(10, 3)
input = autograd.Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]])).transpose(0, 1).contiguous()
inp_emb = embedding(input)
inp_emb.data.shape
packed = torch.nn.utils.rnn.pack_padded_sequence(inp_emb, [4, 3])
torch.nn.utils.rnn.pad_packed_sequence(packed)[0].transpose(0, 1)


# %%
from gensim.models.wrappers import FastText
w2v_model = FastText.load_fasttext_format('data/fasttext/wiki.de.bin')

# %%

# %%
import numpy as np
d = dict()
with open('data/fasttext/wiki.de.vec') as f:
    print(next(f))
    for i in f:
        cur_line = i.split()
        word = cur_line[0]
        # print(cur_line)
        try:
            emb = np.array(list(map(float, cur_line[1:])))
        except:
            print(cur_line)
        d[word] = emb
# %%
# d['hier']
print('lol')
strng
text_splited = text.split('\n')
len(text_splited[0])
print(len(text_splited[0].split()))

# %%
vocab_path="data/nmt_iwslt/vocab.bin"
train_batch_size = 80 # that was 256
eval_batch_size = 64
test_batch_size = 64
bg = BatchGenerator(vocab_path=vocab_path, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, test_batch_size=test_batch_size)

# embedding = nn.Embedding(embeddings.size(0), embeddings.size(1))
len(d)
# d.pop('world')
# %%
embeddings = np.zeros((len(bg.vocab.src), 300))
# %%
new_dict = dict()
for i in bg.vocab.src.word2id.keys():
    if i in d:
        new_dict[i] = d[i]
        embeddings[bg.vocab.src.word2id[i]] = d[i]
len(new_dict.keys())
