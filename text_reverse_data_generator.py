import numpy as np

target_chunk_emb = None
text = []
with open('./data/text8') as f:
    for line in f:
        text.append(line)
    text = ''.join(text)
vocab_size = len(set(text)) + 1
EOS_id = 0

def char_to_id(char):
    if char == ' ':
        return 1
    else:
        return ord(char) - ord('a') + 2

def id_to_char(i):
    if i == 0:
        return ''
    elif i == 1:
        return ' '
    else:
        return chr(ord('a') + i - 2)

def get_batch(text, chunk_length, batch_size):
    while True:
        chunk_starts = np.random.randint(len(text) - chunk_length, size=batch_size)
        yield np.array([text[chunk_start:chunk_start + chunk_length] for chunk_start in chunk_starts])

def revert_words(chunk_batch):
    rev_chunk_batch = []
    for chunk in chunk_batch:
        words = np.split(chunk, np.flatnonzero(chunk == char_to_id(' ')))
        rev_chunk = np.hstack([words[0][::-1]] + [np.roll(word[::-1], 1) for word in words[1:]])
        rev_chunk_batch.append(rev_chunk)
    return np.array(rev_chunk_batch)

def get_data_generators(train_batch_size, chunk_length, eval_batch_size):
    text_ids = {'full': np.array(list(map(char_to_id, text)))}
    text_length = len(text)
    text_ids['train'] = text_ids['full'][:int(text_length * 0.8)]
    text_ids['eval'] = text_ids['full'][int(text_length * 0.8):int(text_length * 0.9)]
    text_ids['decode'] = text_ids['full'][int(text_length * 0.9):]
    train_batch_gen = get_batch(text_ids['train'], batch_size=train_batch_size, chunk_length=chunk_length)
    eval_batch_gen = get_batch(text_ids['eval'], batch_size=eval_batch_size, chunk_length=chunk_length)
    return train_batch_gen, eval_batch_gen
