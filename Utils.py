import torch
import torch.autograd as autograd
import sys
from NetworkConfig import  NetworkConfig

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    # print("vec is ", vec)
    _, idx = torch.max(vec, 0)
    # print("max is ", idx.view(-1).data.tolist()[0])
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    if NetworkConfig.GPU_ID >= 0:
        tensor = tensor.cuda()
    return autograd.Variable(tensor)

    # Compute log sum exp in a numerically stable way for the forward algorithm


def log_sum_exp(vec):
    # print('vec:', vec)
    max_score = vec[argmax(vec)]
    #max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score)))  #max_score_broadcast


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def print_insts(insts):
    print('Instances:')
    for inst in insts:
        print(inst)
    print()