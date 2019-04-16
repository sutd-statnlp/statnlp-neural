import random
from examples.linear_ner.reader import read_insts
from termcolor import colored
from examples.linear_ner.neural import LSTMBuilder
from examples.linear_ner.compiler import NERCompiler
from hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from common.eval import nereval
from hypergraph.NetworkModel import NetworkModel

from hypergraph.Utils import *
START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
PAD = "<PAD>"

if __name__ == "__main__":

    NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING = True ## Always set this to true
    seed = 42
    torch.manual_seed(seed)
    torch.set_num_threads(40) ## for cpu purpose
    np.random.seed(seed)
    random.seed(seed)

    train_file = "data/conll/train.txt.bieos"
    dev_file = "data/conll/dev.txt.bieos"
    test_file = "data/conll/test.txt.bieos"
    trial_file = "data/conll/trial.txt.bieos"


    num_train = 200
    num_dev = 10
    num_test = 10
    num_iter = 200
    device = "cpu" ## cuda:0, cuda:1,
    optimizer_str = "sgd"
    NetworkConfig.NEURAL_LEARNING_RATE = 0.01
    # emb_file = 'data/glove.6B.100d.txt'  ## or None, means random embedding.
    emb_file = None

    NetworkConfig.DEVICE = torch.device(device)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)


    train_insts = read_insts(train_file, True, num_train)
    dev_insts = read_insts(dev_file, False, num_dev)
    test_insts = read_insts(test_file, False, num_test)

    label2id = {START: 0}
    for inst in train_insts:
        for label in inst.output:
            if label not in label2id:
                label2id[label] = len(label2id)
    label2id[STOP] = len(label2id)
    print("label Id Mapping:", label2id)

    max_size = -1
    vocab2id = {}
    char2id = {PAD: 0, UNK: 1}

    labels = [None] *len(label2id)
    for key in label2id:
        labels[label2id[key]] = key

    for inst in train_insts + dev_insts + test_insts:
        max_size = max(len(inst.input), max_size)
        for word in inst.input:
            if word not in vocab2id:
                vocab2id[word] = len(vocab2id)

    numpy_emb = load_emb_glove(emb_file, vocab2id)


    for inst in train_insts:
        max_size = max(len(inst.input), max_size)
        for word in inst.input:
            for ch in word:
                if ch not in char2id:
                    char2id[ch] = len(char2id)


    print(colored('vocab_2id:', 'red'), len(vocab2id))

    chars = [None] * len(char2id)
    for key in char2id:
        chars[char2id[key]] = key

    for inst in train_insts + dev_insts + test_insts:
        max_word_length = max([len(word) for word in inst.input])
        inst.word_seq = torch.LongTensor([vocab2id[word] if word in vocab2id else vocab2id[UNK] for word in inst.input]).to(NetworkConfig.DEVICE)
        char_seq_list = [[char2id[ch] if ch in char2id else char2id[UNK] for ch in word] + [char2id[PAD]] * (max_word_length - len(word)) for word in inst.input]
        inst.char_seq_tensor = torch.LongTensor(char_seq_list).to(NetworkConfig.DEVICE)
        inst.char_seq_len = torch.LongTensor([len(word) for word in inst.input]).to(NetworkConfig.DEVICE)

    evaluator = nereval()

    '''' Training  '''
    gnp = TensorGlobalNetworkParam()
    fm = LSTMBuilder(gnp, len(vocab2id), len(label2id), numpy_emb, char2id, chars, NetworkConfig.DEVICE)
    compiler = NERCompiler(label2id, max_size)
    model = NetworkModel(fm, compiler, evaluator)
    model.learn(train_insts, num_iter, dev_insts, test_insts, optimizer_str)
    '''' End of Training  '''
    ''' Test  '''
    model.load_state_dict(torch.load('best_model.pt'))
    results = model.test(test_insts)

    # to get the prediction for each instance: use: inst.get_prediction()
    ret = model.evaluator.eval(test_insts)
    print(ret)


