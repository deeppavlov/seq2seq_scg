# Taken from:
# https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5

import torch
import torch.autograd as autograd
import torch.nn.functional as F


def sample_gumbel(input):
    noise = input.new(input.size()).uniform_()
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return autograd.Variable(noise)

def gumbel_softmax_sample(input, temperature=1.0, hard=False):
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    x = F.softmax(x)

    if hard:
        max_val, _ = torch.max(x, x.dim()-1)
        x_hard = x == max_val.unsqueeze(-1).expand_as(x)
        tmp = (x_hard.float() - x)
        tmp2 = tmp.clone()
        tmp2.detach_()
        x = tmp2 + x

    return x.view_as(input)
