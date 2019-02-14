from hypergraph.NetworkCompiler import NetworkCompiler
from hypergraph.NetworkIDMapper import NetworkIDMapper
from hypergraph.TensorBaseNetwork import TensorBaseNetwork
from hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from hypergraph.NeuralBuilder import NeuralBuilder
from hypergraph.NetworkModel import NetworkModel
import torch.nn as nn
from hypergraph.Utils import *
from common.LinearInstance import LinearInstance
from common.eval import semieval
import re
from termcolor import colored
import random

class TagNetworkCompiler(NetworkCompiler):

    def __init__(self, label_map, max_size=20, max_seg_size=5):
        super().__init__()
        self.labels = ["x"] * len(label_map)
        self.label2id = label_map

        for key in self.label2id:
            self.labels[self.label2id[key]] = key
        print(self.labels)
        #print("Inside compiler: ", self.labels)
        NetworkIDMapper.set_capacity(np.asarray([200, 200, 100, 4], dtype=np.int64))

        # print(self.label2id)
        # print(self.labels)
        self._all_nodes = None
        self._all_children = None
        self._max_size = max_size
        self._max_seg_size = max_seg_size
        print("The max size: ", self._max_size)

        print("Building generic network..")
        self.build_generic_network()

    def to_root(self, size):
        return self.to_node(0, size - 1, len(self.labels) - 1, 3)

    def to_tag(self, start_pos, end_pos, label_id):
        return self.to_node(start_pos, end_pos, label_id, 1)

    def to_tag_prime(self, end_pos, label_id):
        return self.to_node(0, end_pos, label_id, 2)

    def to_leaf(self, ):
        return self.to_node(0, 0, 0, 0)

    def to_node(self, start, end, label_id, node_type):
        length = end - start + 1
        return NetworkIDMapper.to_hybrid_node_ID(np.asarray([end, length, label_id, node_type]))

    def compile_labeled(self, network_id, inst, param):

        builder = TensorBaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        builder.add_node(leaf)

        output = inst.get_output() ### list of spans, span is tuple.
        children = [leaf]
        for span_tuple in output:
            left = span_tuple[0]
            right = span_tuple[1]
            label = span_tuple[2]
            tag_node = self.to_tag(left, right, self.label2id[label])
            tag_prime = self.to_tag_prime(right, self.label2id[label])
            builder.add_node(tag_node)
            builder.add_node(tag_prime)
            builder.add_edge(tag_node, children)
            builder.add_edge(tag_prime, [tag_node])
            children = [tag_prime]
        root = self.to_root(inst.size())
        builder.add_node(root)
        builder.add_edge(root, children)
        network = builder.build(network_id, inst, param, self)
        return network

    def compile_unlabeled_old(self, network_id, inst, param):
        # return self.compile_labeled(network_id, inst, param)
        builder = TensorBaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        builder.add_node(leaf)

        children = [leaf]
        for i in range(inst.size()):
            current = [None for k in range(len(self.labels))]
            for l in range(len(self.labels)):
                tag_node = self.to_tag(i, l)
                builder.add_node(tag_node)
                for child in children:
                    builder.add_edge(tag_node, [child])
                current[l] = tag_node

            children = current
        root = self.to_root(inst.size())
        builder.add_node(root)
        for child in children:
            builder.add_edge(root, [child])
        network = builder.build(network_id, inst, param, self)
        return network

    def compile_unlabeled(self, network_id, inst, param):
        builder = TensorBaseNetwork.NetworkBuilder.builder()
        root_node = self.to_root(inst.size())
        all_nodes = self._all_nodes
        root_idx = np.argwhere(all_nodes == root_node)[0][0]
        node_count = root_idx + 1
        network = builder.build_from_generic(network_id, inst, self._all_nodes, self._all_children, node_count, self.num_hyperedge, param, self)
        return network


    def build_generic_network(self, ):

        builder = TensorBaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        builder.add_node(leaf)

        children = [leaf]
        for i in range(self._max_size):
            for j in range(i, i + self._max_seg_size -1):
                for l in range(len(self.labels)):
                    if l ==0 or l == len(self.labels) - 1:
                        continue
                    if i!= j and l == self.label2id["O"]:
                        continue
                    span_node = self.to_tag(i, j, l)
                    added = False
                    for prev_label_id in range(len(self.labels)):
                        if i > 0:
                            child_node = self.to_tag_prime(i-1, prev_label_id)
                            if builder.contains_node(child_node):
                                added = True
                                builder.add_node(span_node)
                                builder.add_edge(span_node, [child_node])
                        else:
                            added = True
                            builder.add_node(span_node)
                            builder.add_edge(span_node, [leaf])
                    if added:
                        end_prime_node = self.to_tag_prime(j, l)
                        builder.add_node(end_prime_node)
                        builder.add_edge(end_prime_node, [span_node])

                        root = self.to_root(j+1)
                        builder.add_node(root)
                        builder.add_edge(root, [end_prime_node])
        self._all_nodes, self._all_children, self.num_hyperedge = builder.pre_build()

    def decompile(self, network):
        inst = network.get_instance()

        size = inst.size()
        root_node = self.to_root(size)
        all_nodes = network.get_all_nodes()
        curr_idx = np.argwhere(all_nodes == root_node)[0][0] #network.count_nodes() - 1 #self._all_nodes.index(root_node)
        prediction = []
        while curr_idx != 0:
            children = network.get_max_path(curr_idx)
            child = children[0]
            child_arr = network.get_node_array(child)
            type = child_arr[3]
            if type == 1:
                end = child_arr[0]
                start =  end - child_arr[1] + 1
                label = self.labels[child_arr[2]]
                prediction.append((start, end, label))
                # prediction[size - i - 1] = self.labels[child_arr[1]]

            curr_idx = child
        # print(prediction)
        inst.set_prediction(prediction[::-1])
        return inst


class TagNeuralBuilder(NeuralBuilder):
    def __init__(self, gnp, voc_size, label_size, char2id, chars, char_emb_size, charlstm_hidden_dim, lstm_hidden_size = 100, dropout = 0.5):
        super().__init__(gnp)
        self.token_embed = 100
        self.label_size = label_size
        print("vocab size: ", voc_size)
        # self.word_embed = nn.Embedding(voc_size, self.token_embed, padding_idx=0).to(NetworkConfig.DEVICE)
        self.word_embed = nn.Embedding(voc_size, self.token_embed).to(NetworkConfig.DEVICE)


        self.char_emb_size = char_emb_size
        if char_emb_size > 0:
            from features.char_lstm import CharBiLSTM
            self.char_bilstm = CharBiLSTM(char2id, chars, char_emb_size, charlstm_hidden_dim).to(NetworkConfig.DEVICE)
            # self.char_embeddings = nn.Embedding(len(char2id), self.char_emb_size).to(NetworkConfig.DEVICE)
            # self.char_rnn = nn.LSTM(self.char_emb_size, charlstm_hidden_dim//2, batch_first=True, bidirectional=True).to(NetworkConfig.DEVICE)
        self.word_drop = nn.Dropout(dropout).to(NetworkConfig.DEVICE)


        lstm_input_size = self.token_embed + charlstm_hidden_dim

        self.rnn = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True,bidirectional=True).to(NetworkConfig.DEVICE)
        # self.linear = nn.Linear(lstm_hidden_size * 2, label_size).to(NetworkConfig.DEVICE)
        label_hidden_size = 100
        self.f_label = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, label_hidden_size).to(NetworkConfig.DEVICE),
            nn.ReLU().to(NetworkConfig.DEVICE),
            nn.Linear(label_hidden_size, label_size).to(NetworkConfig.DEVICE)
        ).to(NetworkConfig.DEVICE)

    def load_pretrain(self, path, word2idx):
        emb = load_emb_glove(path, word2idx, self.token_embed)
        self.word_embed.weight.data.copy_(torch.from_numpy(emb))
        self.word_embed = self.word_embed.to(NetworkConfig.DEVICE)

    # @abstractmethod
    # def extract_helper(self, network, parent_k, children_k, children_k_index):
    #     pass
    def build_nn_graph(self, instance):

        word_seq = instance.word_seq
        word_vec = self.word_embed(word_seq).unsqueeze(0) ###1 x sent_len x embedding size.
        word_rep = [word_vec]

        if self.char_emb_size > 0:
            char_seq_tensor = instance.char_seq_tensor.unsqueeze(0)
            char_seq_len = instance.char_seq_len.unsqueeze(0)
            char_features = self.char_bilstm.get_last_hiddens(char_seq_tensor, char_seq_len)  # batch_size, sent_len, char_hidden_dim
            word_rep.append(char_features)

        word_rep = torch.cat(word_rep, 2)
        word_rep = self.word_drop(word_rep)
        #
        lstm_out, _ = self.rnn(word_rep, None)
        lstm_out = lstm_out.squeeze(0) # sent_len * hidden_size
        sent_len, lstm_dim = lstm_out.size()

        square = lstm_out.view(sent_len, 1, lstm_dim).expand(sent_len, sent_len, lstm_dim)
        square_t = square.transpose(0, 1)

        ret = square_t - square
        ret = ret[:, 1: sent_len - 1, :]

        half_lstm_dim = lstm_dim // 2

        fwd = ret[:sent_len - 2, :, :half_lstm_dim]

        bwd = ret[2:, :, half_lstm_dim:].transpose(0, 1)

        bi = torch.cat([fwd, bwd], 2)
        spans = self.f_label(bi)

        zeros = torch.zeros((sent_len - 2, sent_len - 2, 1)).to(NetworkConfig.DEVICE)  # score of (), empty label
        spans = torch.cat([zeros, spans], 2)

        spans[sent_len - 3, 0, 0] = 0
        return spans

    # def get_nn_score(self, network, parent_k):
    #     parent_arr = network.get_node_array(parent_k)  # pos, label_id, node_type
    #     pos = parent_arr[0]
    #     label_id = parent_arr[1]
    #     node_type = parent_arr[2]
    #
    #     if node_type == 0 or node_type == 2: #Start, End
    #         return torch.tensor(0.0).to(NetworkConfig.DEVICE)
    #     else:
    #         nn_output = network.nn_output
    #         return nn_output[pos][label_id]


    def get_label_id(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        return parent_arr[2]

    def build_node2nn_output(self, network):
        num_nodes = network.count_nodes()
        nodeid2nn = [0] * num_nodes
        for k in range(num_nodes):
            parent_arr = network.get_node_array(k)  # pos, label_id, node_type
            size = network.get_instance().size()
            right, length, node_type, label_id = parent_arr
            left = right - length + 1  ## right is inclusive

            if node_type != 1: ### not label node.
                idx = (size - 1) * size  ## a index with 0
            else:
                row = left * size + right
                idx = row * self.label_size + label_id
            nodeid2nn[k] = idx
        return nodeid2nn



class TagReader():
    label2id_map = {}
    label2id_map["<START>"] = 0

    Stats = {'MAX_WORD_LENGTH':0}

    @staticmethod
    def read_insts(file, is_labeled, number):
        insts = []
        inputs = []
        outputs = []
        f = open(file, 'r', encoding='utf-8')
        start = 0
        end = 0
        pos = 0
        for line in f:
            line = line.strip()
            if len(line) == 0:
                inst = LinearInstance(len(insts) + 1, 1, inputs, outputs)
                if is_labeled:
                    inst.set_labeled()
                else:
                    inst.set_unlabeled()
                insts.append(inst)
                inputs = []
                outputs = []
                start = 0
                end = 0
                pos = 0
                if len(insts) >= number and number > 0:
                    break

            else:
                fields = line.split()
                input = fields[0]
                input = re.sub('\d', '0', input)
                output = fields[-1]

                TagReader.Stats['MAX_WORD_LENGTH'] = max(TagReader.Stats['MAX_WORD_LENGTH'], len(input))

                if output == "O":
                    label = "O"
                else:
                    label = output[2:]
                if not label in TagReader.label2id_map:
                    TagReader.label2id_map[label] = len(TagReader.label2id_map)

                if label == "O":
                    start = pos
                    end = pos
                    outputs.append((start, end, label))
                elif output.startswith("E-"):
                    end = pos
                    outputs.append((start, end, label))
                elif output.startswith("B-"):
                    start = pos
                elif output.startswith("S-"):
                    start = pos
                    end = pos
                    outputs.append((start, end, label))
                pos += 1
                inputs.append(input)

        f.close()

        return insts


START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
PAD = "<PAD>"

if __name__ == "__main__":

    NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH = True
    NetworkConfig.IGNORE_TRANSITION = False
    # NetworkConfig.GPU_ID = -1
    # NetworkConfig.ECHO_TRAINING_PROGRESS = -1
    # NetworkConfig.LOSS_TYPE = LossType.SSVM
    NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING = True
    seed = 1234
    torch.manual_seed(seed)
    torch.set_num_threads(40)
    np.random.seed(seed)
    random.seed(seed)



    train_file = "data/conll/train.txt.bieos"
    dev_file = "data/conll/dev.txt.bieos"
    test_file = "data/conll/test.txt.bieos"
    trial_file = "data/conll/trial.txt.bieos"


    TRIAL = False
    num_train = 50
    num_dev = 50
    num_test = 50
    num_iter = 100
    batch_size = 1
    device = "cpu"
    optimizer_str = "adam"
    num_thread = 1
    dev_file = train_file
    test_file = train_file

    char_emb_size= 25
    charlstm_hidden_dim = 50


    if TRIAL == True:
        data_size = -1
        train_file = trial_file
        dev_file = trial_file
        test_file = trial_file

    if device == "gpu":
        NetworkConfig.DEVICE = torch.device("cuda:0")
        torch.cuda.manual_seed(seed)

    if num_thread > 1:
        NetworkConfig.NUM_THREADS = num_thread
        print('Set NUM_THREADS = ', num_thread)

    train_insts = TagReader.read_insts(train_file, True, num_train)
    random.shuffle(train_insts)
    dev_insts = TagReader.read_insts(dev_file, False, num_dev)
    test_insts = TagReader.read_insts(test_file, False, num_test)



    TagReader.label2id_map["<ROOT>"] = len(TagReader.label2id_map)
    print("map:", TagReader.label2id_map)

    max_size = -1
    # vocab2id = {}
    vocab2id = {START: 0, STOP: 1}
    char2id = {PAD: 0, UNK: 1}

    for inst in train_insts + dev_insts + test_insts:
        max_size = max(len(inst.input), max_size)
        for word in inst.input:
            if word not in vocab2id:
                vocab2id[word] = len(vocab2id)

                for ch in word:
                    if ch not in char2id:
                        char2id[ch] = len(char2id)


    print(colored('vocab_2id:', 'red'), len(vocab2id))

    chars = [None] * len(char2id)
    for key in char2id:
        chars[char2id[key]] = key

    max_word_length = TagReader.Stats['MAX_WORD_LENGTH']
    print(colored('MAX_WORD_LENGTH:', 'blue'), TagReader.Stats['MAX_WORD_LENGTH'])


    for inst in train_insts + dev_insts + test_insts:
        inst.word_seq = torch.tensor([vocab2id[START]] + [vocab2id[word] for word in inst.input] + [vocab2id[STOP]]).to(NetworkConfig.DEVICE)
        char_seq_list = [[char2id[ch] for ch in word] + [char2id[PAD]] * (max_word_length - len(word)) for word in inst.input]
        char_seq_list = [[char2id[PAD]] * max_word_length] + char_seq_list + [[char2id[PAD]] * max_word_length]
        # char_seq_list = [torch.tensor([char2id[ch] for ch in word]).to(NetworkConfig.DEVICE)  for word in inst.input]
        # inst.char_seqs = char_seq_list
        inst.char_seq_tensor = torch.tensor(char_seq_list).to(NetworkConfig.DEVICE)
        # char_seq_tensor: (1, sent_len, word_length)
        inst.char_seq_len = torch.tensor([max_word_length] + [len(word) for word in inst.input] + [max_word_length]).to(NetworkConfig.DEVICE)



    gnp = TensorGlobalNetworkParam()
    fm = TagNeuralBuilder(gnp, len(vocab2id), len(TagReader.label2id_map), char2id, chars, char_emb_size, charlstm_hidden_dim,)
    # fm.load_pretrain('data/glove.6B.100d.txt', vocab2id)
    fm.load_pretrain(None, vocab2id)
    print(list(TagReader.label2id_map.keys()))
    compiler = TagNetworkCompiler(TagReader.label2id_map, max_size)


    evaluator = semieval()
    model = NetworkModel(fm, compiler, evaluator)
    # model.check_every = 2000




    if batch_size == 1:
        model.learn(train_insts, num_iter, dev_insts, test_insts, optimizer_str, batch_size)
    else:
        model.learn_batch(train_insts, num_iter, dev_insts, batch_size)

    model.load_state_dict(torch.load('best_model.pt'))



    results = model.test(test_insts)
    for inst in results:
        print(inst.get_input())
        print(inst.get_output())
        print(inst.get_prediction())
        print()

    ret = model.evaluator.eval(test_insts)
    print(ret)


