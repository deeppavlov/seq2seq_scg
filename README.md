<h1>Seq2Seq</h1>

<h2> Requirements </h2>
We strongly recommend use anaconda, as the easiest way to setting up requirements.

1) pytorch

... to be continued
<h2>Setting up</h2>
1) Place IWSLT german->english dataset in the directory:

data/nmt_iwslt


2) In order to create vocab.bin run this command:

python vocab.py --train_src data/nmt_iwslt/train.de-en.de --train_tgt data/nmt_iwslt/train.de-en.en --output data/nmt_iwslt/vocab.bin


<!-- cp vocab.bin data/nmt_iwslt/vocab.bin

rm vocab.bin -->

2) Download pretrained word vectors:

cd data/fasttext

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec

3)
(TODO: create script with argparse output param)

To create pickle files for pretrained embeddings run(they are just numpy arrays):


cp my_de_emb data/fasttext/my_de_emb

cp my_en_emb data/fasttext/my_en_emb

<h2> Usage </h2>
1) Run training

runner_cuda.sh gpu_id

for example:

 runner_cuda.sh 0

to run on the gpu with id 0.

2)

for now one can only evaluate model on eval(not on the test)
run this commands:

python BLEU_scorer.py

python bleu_wrapper.py
