import os


NOT_AVAILABLE = 'NA'

def process_train_params(params):
    if params['start_model'] is not None:
        if params['start_step'] in ['first', 'last']:
            first_step, last_step = None, None
            for file_name in os.listdir(os.path.join('./saved_models', params['start_model'])):
                try:
                    step = int(file_name.split('=')[1])
                    if first_step is None or step < first_step:
                        first_step = step
                    if last_step is None or step > last_step:
                        last_step = step
                except:
                    pass
            if first_step is None:
                raise ValueError('No saved models in specified path')
            if params['start_step'] == 'first':
                params['start_step'] = first_step
            elif params['start_step'] == 'last':
                params['start_step'] = last_step
        elif not isinstance(params['start_step'], int):
            raise ValueError("'start_step' should be 'first', 'last' or int")
        params['start_model'] = os.path.join(params['start_model'], 'step=' + str(params['start_step']))
    else:
        params['start_step'] = 0
    if params['do_tgt_reverse']:
        raise NotImplementedError('Target reversion is not currently implemented')
    if not params['use_masked_loss']:
        raise ValueError('use_masked_loss = False is deprecated')
    return params

def get_train_params():
    train_params = {
        "train_batch_size": 128,
        "eval_batch_size": 64,
        "test_batch_size": 64,
        "decode_batch_size": 8,
        "vocab_path": "data/nmt_iwslt/vocab.bin",
        "do_tgt_reverse": False, # this flag is ignored!
        "chunk_length": 32,
        "use_masked_loss": True,
        "num_runs": 1,
        "num_steps": 100000,
        "print_skip": 100,
        "save_per_step": 1000,
        "start_model": 'opt_goal=BLEU\
__output=sampling__output_baseline=argmax\
__feed=same__feed_baseline=NA__attn=soft__attn_baseline=NA\
__tf_ratio=(0.0, 0.0)__softmax_t=NA__init_lr=0.0001', # mode_name or None
#        "start_model": None,
        "start_step": 'last', # 'first', 'last' or step number
        "init_lr": 0.001
    }
    return process_train_params(train_params)

def process_model_params(params):
    if params["opt_goal"] not in ['BLEU', 'log-likelihood']:
        print("Unsupported opt_goal: '{}'".format(params["opt_goal"]))
    if params["output_mode"] not in ['argmax', 'sampling']:
        print("Unsupported output_mode: '{}'".format(params["output_mode"]))
    if params["feed_mode"] not in ['argmax', 'sampling', 'softmax', 'gumbel', 'gumbel-st', 'same']:
        print("Unsupported feed_mode: '{}'".format(params["feed_mode"]))
    if params["tf_ratio_range"][0] < 0 or params["tf_ratio_range"][0] > 1:
        print("tf_ratio_range should be a tuple of numbers in range [0, 1]")
    if params["tf_ratio_range"][1] < 0 or params["tf_ratio_range"][1] > 1:
        print("tf_ratio_range should be a tuple of numbers in range [0, 1]")
    if params["feed_baseline"] not in ['no-reinforce', 'argmax', 'av_loss']:
        print("Unsupported feed_baseline: '{}'".format(params["feed_baseline"]))
    if params["output_baseline"] not in ['argmax', 'av_loss']:
        print("Unsupported output_baseline: '{}'".format(params["output_baseline"]))
    if params["attention_mode"] not in ['soft', 'gumbel', 'gumbel-st', 'hard', 'argmax']:
        print("Unsupported attention_mode: '{}'".format(params["attention_mode"]))
    if params["attention_baseline"] not in ['argmax']:
        print("Unsupported attention_baseline: '{}'".format(params["attention_baseline"]))

    if params["opt_goal"] != 'log-likelihood':
        params["output_mode"] = 'sampling'
        params["feed_mode"] = 'same'
    if not(
        params["feed_mode"] == 'sampling' or
        params["output_mode"] == 'sampling' and
        params["feed_mode"] == 'same' and
        params["opt_goal"] == 'log-likelihood'
    ):
        params['feed_baseline'] = NOT_AVAILABLE
    if params["feed_mode"] in ['argmax', 'sampling', 'same']:
        params['softmax_t_range'] = NOT_AVAILABLE
    if params["opt_goal"] != 'BLEU':
        params["output_baseline"] = NOT_AVAILABLE
    if params["opt_goal"] == 'BLEU':
        params["baseline_output_mode"] = 'sampling'
    if params["feed_baseline"] == 'argmax' or params["output_baseline"] == 'argmax':
        params['baseline_feed_mode'] = 'argmax'
    if params["attention_mode"] != 'hard':
        params['attention_baseline'] = NOT_AVAILABLE
    if params["attention_baseline"] == 'argmax':
        params['baseline_attention_mode'] = 'argmax'

def get_model_params():
    model_params = {
        "opt_goal": 'BLEU',
        "output_mode": 'argmax',
        "feed_mode": 'sampling',
        "tf_ratio_range": (0.0, 0.0),
        "feed_baseline": 'argmax',
        "softmax_t_range": (1.0, 0.01),
        "output_baseline": 'argmax',
        "baseline_output_mode": 'argmax',
        "baseline_feed_mode": None,
        "attention_mode": 'soft',
        "attention_baseline": 'argmax',
        "baseline_attention_mode": None
    }
    process_model_params(model_params)
    return model_params

