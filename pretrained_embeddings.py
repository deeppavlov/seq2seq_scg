import numpy as np
from data_generator import BatchGenerator
import pickle
from vocab import Vocab, VocabEntry
import torch.nn as nn
import torch
# %%
vocab_path="data/nmt_iwslt/vocab.bin"
train_batch_size = 80 # that was 256
eval_batch_size = 64
test_batch_size = 64
bg = BatchGenerator(vocab_path=vocab_path, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, test_batch_size=test_batch_size)
# %%

def load_and_save(fasttext_file, output_file, vcb):
    """ load fasttext .vec file and perform saving in <output_file> pickle file """
    d = dict()
    with open(fasttext_file) as f:
        print(next(f))
        for i in f:
            cur_line = i.split()
            word = cur_line[0]
            try:
                emb = np.array(list(map(float, cur_line[1:])))
            except:
                print('this line is not in format:')
                print(cur_line)
            d[word] = emb
    new_dict = dict()
    embeddings = np.zeros((len(vcb), 300))
    print ('Creating embeddings with shape: ', embeddings.shape)
    for i in vcb.keys():
        if i in d:
            new_dict[i] = d[i]
            embeddings[vcb[i]] = d[i]
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)

# %%
load_and_save('data/fasttext/wiki.de.vec', 'my_de_emb', bg.vocab.src.word2id)
# %%
load_and_save('data/fasttext/wiki.en.vec', 'my_en_emb', bg.vocab.tgt.word2id)
