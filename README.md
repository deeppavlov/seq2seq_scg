<h1>Seq2Seq Neural Machine Translation model with different optimization techniques</h1>

This is our implementation of seq2seq neural machine translation model with different optimization techniques. For details, see the corresponding [NIPS'17 workshop paper](https://www.overleaf.com/read/ncpvyxbhhjgq)

<h2>Setting up</h2>

1) Get [IWSLT'14 german->english dataset](https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz).
Then, place it in the directory:
```
data/nmt_iwslt
```


2) In order to create vocab.bin run this command:
```shell
$ python utils/vocab.py --train_src data/nmt_iwslt/train.de-en.de --train_tgt data/nmt_iwslt/train.de-en.en --output data/nmt_iwslt/vocab.bin
```


3) Download pretrained word vectors:
```shell
$ mkdir data/fasttext
$ cd data/fasttext
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
```


<h2> Usage </h2>

1) Specify train and model parameters in ```utils/param_provider.py```

2) Run training

To train on CPU just execute ```train.py```:
```shell
$ python train.py
```

To train on GPU:
```shell
$ source train_on_gpu.sh gpu_id
```

for example:
```shell
$ source train_on_gpu.sh 0
```
to run on GPU with id 0.

By default, the script ```train.py``` do not evaluates the model during training. If you want to evaluate, run it with key ```--do_eval```:
```shell
$ python train.py --do_eval
```

Or, run the following script to train and eval on GPU:
```shell
$ source train_and_eval_on_gpu.sh gpu_id
```

3) Evaluate your model on test dataset

First, specify the model you want to evaluate as a 'start_model' in ```utils/param_provider.py```.

Then, run the previously trained model in inference mode:
```shell
$ python inference.py
```
Or, to infer on GPU:
```shell
$ source inference_on_gpu.sh gpu_id
```

Last, run this command to evaluate BLEU score:
```shell
$ python compute_bleu.py
```
