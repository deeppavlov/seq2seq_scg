<h1>Seq2Seq Neural Machine Translation model with different optimization techniques</h1>

This is our implementation of seq2seq neural machine translation model with different optimization techniques. For details, see the corresponding NIPS'17 workshop paper https://www.overleaf.com/read/ncpvyxbhhjgq

<h2> Requirements </h2>
We recommend to use anaconda, as the easiest way for setting up requirements.

1) python 3.6
2) pytorch 0.2

<h2>Setting up</h2>

1) Get IWSLT'14 german->english dataset https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz
Then place it in the directory:
```
data/nmt_iwslt
```


2) In order to create vocab.bin run this command:
```shell
$ python vocab.py --train_src data/nmt_iwslt/train.de-en.de --train_tgt data/nmt_iwslt/train.de-en.en --output data/nmt_iwslt/vocab.bin
```


3) Download pretrained word vectors:
```shell
$ mkdir data/fasttext
$ cd data/fasttext
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
```


<h2> Usage </h2>

1) Specify train and model parameters in ```param_provider.py```

2) Run training

To train on CPU just execute ```notebook.py```:
```shell
$ python notebook.py
```

To train on GPU:
```shell
$ source runner_cuda.sh gpu_id
```

for example:
```shell
$ source runner_cuda.sh 0
```

to run on GPU with id 0.

3) Evaluate model on test dataset

First, specify the model you want to evaluate as a 'start_model' in ```param_provider.py```
Then run these two commands in order to evaluate BLEU score:
```shell
$ python BLEU_scorer.py
$ python bleu_wrapper.py
```
