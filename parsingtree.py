from hypergraph.NetworkCompiler import NetworkCompiler
from hypergraph.NetworkIDMapper import NetworkIDMapper
from hypergraph.TensorBaseNetwork import TensorBaseNetwork
from hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from hypergraph.NeuralBuilder import NeuralBuilder
from hypergraph.NetworkModel import NetworkModel
import torch.nn as nn
from hypergraph.Utils import *
from common.TreeInstance import TreeInstance
from common.eval import constituent_eval
import re
from termcolor import colored
from enum import Enum
import examples.parsingtree.trees as trees
import functools



START_IDX = 0 #"<START>"
STOP_IDX = 1 #"<STOP>"
UNK_IDX = 2 #"<UNK>"

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"


class NodeType(Enum):
    sink = 0
    leaf = 1
    label = 2
    span_prime = 3
    span = 4
    root = 5


class TreeNetworkCompiler(NetworkCompiler):

    def __init__(self, label_map, max_size=200):
        super().__init__()
        self.labels = [()] * (len(label_map))  ##  include (), but not dummy label
        self.label2id = label_map

        for key in self.label2id:
            self.labels[self.label2id[key]] = key


        self.max_size = max_size
        ### Length, rightIdx, nodeType, LabelId
        NetworkIDMapper.set_capacity(
            np.asarray([self.max_size, self.max_size, NodeType.root.value + 1, len(self.labels) + 1], dtype=np.int64))

        print(self.label2id)
        print(self.labels)
        self._all_nodes = None
        self._all_children = None

        # self.build_generic_network()

    def to_sink(self):
        return self.to_node(0, 0, NodeType.sink.value, 0)

    def to_leaf(self, left_idx):
        return self.to_node(1, left_idx + 1, NodeType.leaf.value, 0)

    def to_label(self, left_idx, right_idx, label_id):
        return self.to_node(right_idx - left_idx, right_idx, NodeType.label.value, label_id)

    def to_span_prime(self, left_idx, right_idx):
        return self.to_node(right_idx - left_idx, right_idx, NodeType.span_prime.value, 0)

    def to_span(self, left_idx, right_idx):
        return self.to_node(right_idx - left_idx, right_idx, NodeType.span.value, 0)

    def to_root(self, size):
        return self.to_node(size, size, NodeType.root.value, 0)

    def to_node(self, length, right_idx, node_type, label_id):
        return NetworkIDMapper.to_hybrid_node_ID(np.asarray([length, right_idx, node_type, label_id]))

    def compile_labeled(self, network_id, inst, param):

        builder = TensorBaseNetwork.NetworkBuilder.builder()

        node_sink = self.to_sink()
        builder.add_node(node_sink)

        node_root = self.to_root(inst.size())
        builder.add_node(node_root)

        size = inst.size()

        gold = inst.get_output()

        for length in range(1, size + 1):
            for left in range(0, size + 1 - length):
                right = left + length

                if length == 1:
                    node_leaf = self.to_leaf(left)
                    builder.add_node(node_leaf)
                    builder.add_edge(node_leaf, [node_sink])


                node_span = self.to_span(left, right)
                builder.add_node(node_span)

                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.label2id[oracle_label]

                if length > 1:
                    node_span_prime = self.to_span_prime(left, right)
                    builder.add_node(node_span_prime)

                label = oracle_label
                node_label = self.to_label(left, right, self.label2id[label])
                builder.add_node(node_label)


                if length > 1:
                    builder.add_edge(node_label, [node_sink])
                    builder.add_edge(node_span, [node_label, node_span_prime])
                    oracle_splits = gold.oracle_splits(left, right)
                    oracle_split = min(oracle_splits)
                    k = oracle_split

                    left_child = self.to_span(left, k)
                    right_child = self.to_span(k, right)
                    if builder.contains_node(left_child) and builder.contains_node(right_child):
                        builder.add_edge(node_span_prime, [left_child, right_child])

                else:

                    node_leaf = self.to_leaf(left)

                    if label:
                        builder.add_edge(node_span, [node_label])
                        builder.add_edge(node_label, [node_leaf])
                    else:

                        builder.add_edge(node_span, [node_leaf])


                if length == size:
                    builder.add_edge(node_root, [node_span])

        network = builder.build(network_id, inst, param, self)
        return network

    def compile_unlabeled(self, network_id, inst, param):
        # return self.compile_labeled(network_id, inst, param)
        builder = TensorBaseNetwork.NetworkBuilder.builder()

        node_sink = self.to_sink()
        builder.add_node(node_sink)

        node_root = self.to_root(inst.size())
        builder.add_node(node_root)

        size = inst.size()



        for length in range(1, size + 1):
            for left in range(0, size + 1 - length):
                right = left + length

                if length == 1:
                    node_leaf = self.to_leaf(left)
                    builder.add_node(node_leaf)
                    builder.add_edge(node_leaf, [node_sink])

                node_span = self.to_span(left, right)
                builder.add_node(node_span)

                if length > 1:
                    node_span_prime = self.to_span_prime(left, right)
                    builder.add_node(node_span_prime)

                for label in self.labels:
                    if length == size and label == ():
                        continue

                    node_label = self.to_label(left, right, self.label2id[label])
                    builder.add_node(node_label)

                    if length > 1:
                        builder.add_edge(node_label, [node_sink])
                        builder.add_edge(node_span, [node_label, node_span_prime])
                    else: #length == 1
                        builder.add_edge(node_span, [node_label])

                        node_leaf = self.to_leaf(left)

                        builder.add_edge(node_label, [node_leaf])
                        builder.add_edge(node_span, [node_leaf])


                for k in range(left + 1, right):

                    left_child = self.to_span(left, k)
                    right_child = self.to_span(k, right)
                    if builder.contains_node(left_child) and builder.contains_node(right_child):
                        builder.add_edge(node_span_prime, [left_child, right_child])

                if length == size:
                    builder.add_edge(node_root, [node_span])

        network = builder.build(network_id, inst, param, self)
        return network

    # def build_generic_network(self, ):
    #
    #     network = builder.build(None, None, None, None)
    #     self._all_nodes = network.get_all_nodes()
    #     self._all_children = network.get_all_children()

    def decompile(self, network):
        inst = network.get_instance()

        size = inst.size()
        root_node = self.to_root(size)
        all_nodes = network.get_all_nodes()
        root_idx = np.argwhere(all_nodes == root_node)[0][0]  # network.count_nodes() - 1 #self._all_nodes.index(root_node)

        children = network.get_max_path(root_idx)  # children[0]: root node
        prediction = self.to_tree_helper(network, children[0])
        pred_tree = prediction[0].convert()
        inst.set_prediction(pred_tree)
        return inst

    def to_tree_helper(self, network, curr_idx):

        node_arr = network.get_node_array(curr_idx)

        children = network.get_max_path(curr_idx)

        label_node_arr = network.get_node_array(children[0])
        left_idx = label_node_arr[1] - label_node_arr[0]
        label_idx = label_node_arr[3]
        label = self.labels[label_idx]

        if children[1] != network.non_exist_node_id:
            span_prime_node = children[1]

            split_span_nodeses = network.get_max_path(span_prime_node)

            left_span_node, right_span_node = split_span_nodeses[0], split_span_nodeses[1]

            left_tree_children = self.to_tree_helper(network, left_span_node)
            right_tree_children = self.to_tree_helper(network, right_span_node)

            tree_children = left_tree_children + right_tree_children

            if label:
                parse_node = trees.InternalParseNode(label, tree_children)
                return [parse_node]
            else:
                return  tree_children


        else:
            inst = network.get_instance()
            word, tag = inst.get_input()[left_idx]
            parse_node = trees.LeafParseNode(left_idx, tag, word)

            if label:
                parse_node = trees.InternalParseNode(label, [parse_node])

            return [parse_node]



class TreeNeuralBuilder(NeuralBuilder):
    def __init__(self, gnp, voc_size, word_embed_dim, tag_size, tag_embed_dim, lstm_dim = 250, label_hidden_size = 250, dropout = 0.4):
        super().__init__(gnp)
        # self.word_embed = nn.Embedding(voc_size, self.token_embed, padding_idx=0).to(NetworkConfig.DEVICE)
        self.word_embed_dim = word_embed_dim
        self.lstm_dim = lstm_dim
        self.word_embeddings = nn.Embedding(voc_size, word_embed_dim).to(NetworkConfig.DEVICE)
        self.tag_embeddings = nn.Embedding(tag_size, tag_embed_dim).to(NetworkConfig.DEVICE)

        tag_embed_parameter = np.empty([tag_size, tag_embed_dim])

        scale = np.sqrt(3.0 / tag_embed_dim)
        for i in range(tag_size):
            tag_embed_parameter[i] = np.random.uniform(-scale, scale, [1, tag_embed_dim])

        embed_dim = word_embed_dim + tag_embed_dim
        self.rnn = nn.LSTM(embed_dim, lstm_dim, batch_first=True, bidirectional=True, dropout=dropout).to(NetworkConfig.DEVICE)

        self.f_label = nn.Sequential(
            nn.Linear(lstm_dim * 2, label_hidden_size).to(NetworkConfig.DEVICE),
            nn.ReLU().to(NetworkConfig.DEVICE),
            nn.Linear(label_hidden_size, gnp.label_size - 1).to(NetworkConfig.DEVICE)
        ).to(NetworkConfig.DEVICE)



    def load_pretrain(self, word2idx):
        emb = load_emb_glove(None, word2idx, self.word_embed_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(emb))
        self.word_embeddings = self.word_embeddings.to(NetworkConfig.DEVICE)


    def build_nn_graph(self, instance):


        word_seq = instance.word_seq
        tag_seq = instance.tag_seq

        size = instance.size()

        tag_embs = self.tag_embeddings(tag_seq)
        word_embs = self.word_embeddings(word_seq)

        word_rep = torch.cat([word_embs, tag_embs], 1).unsqueeze(0)

        lstm_outputs, _ = self.rnn(word_rep, None)
        lstm_outputs = lstm_outputs.squeeze(0)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                    lstm_outputs[right][:self.lstm_dim] -
                    lstm_outputs[left][:self.lstm_dim])
            backward = (
                    lstm_outputs[left + 1][self.lstm_dim:] -
                    lstm_outputs[right + 1][self.lstm_dim:])
            return torch.cat([forward, backward], 0)

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            span_emb = get_span_encoding(left, right)
            non_empty_label_scores = self.f_label(span_emb)
            label_vec = torch.cat([torch.tensor([0.0]).to(NetworkConfig.DEVICE), non_empty_label_scores], 0)
            return label_vec


        spans = {}

        for i in range(size):
            for j in range(i + 1, size + 1):
                label_scores = get_label_scores(i, j)
                spans[i,j] = label_scores

        return spans

    def generate_batches(self, train_insts, batch_size):
        '''
        :param instances:
        :param batch_size:
        :return: A list of tuple (input_seqs, network_id_range)
        '''
        pass

    def build_nn_graph_batch(self, batch_input_seqs):
        pass



    def get_nn_score(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)  # pos, label_id, node_type
        length = parent_arr[0]
        right = parent_arr[1]
        left = right - length
        node_type = parent_arr[2]
        label_id = parent_arr[3]


        if node_type != NodeType.label.value:  # Start, End
            return torch.tensor(0.0).to(NetworkConfig.DEVICE)
        else:
            spans = network.nn_output
            return spans[left, right][label_id]

    def get_label_id(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        return parent_arr[3]


class TreeReader():
    @staticmethod
    def read_insts(file, is_labeled, number):
        print('Reading data from ', file, '...')
        gold_trees = trees.load_trees(file)

        insts = []

        for tree in gold_trees:
            leaves = list(tree.leaves())
            inputs = [(leaf.word, leaf.tag) for leaf in leaves]
            inst = TreeInstance(len(insts) + 1, 1, inputs, tree)
            if is_labeled:
                inst.set_labeled()
            else:
                inst.set_unlabeled()
            insts.append(inst)


        if number > -1:
            insts = insts[:number]

        return insts


if __name__ == "__main__":

    torch.manual_seed(1234)
    torch.set_num_threads(40)
    np.random.seed(1234)

    train_file = "data/ptb/02-21.10way.clean"
    dev_file = "data/ptb/22.auto.clean"
    test_file = "data/ptb/23.auto.clean"
    trial_file = "data/ptb/trial.txt"

    TRIAL = False
    num_train = 30
    num_dev = 30
    num_test = 30
    num_iter = 300
    batch_size = 1
    device = "cpu"
    num_thread = 1
    dev_file = test_file

    if TRIAL == True:
        data_size = -1
        train_file = trial_file
        dev_file = trial_file
        test_file = trial_file

    if device == "gpu":
        NetworkConfig.DEVICE = torch.device("cuda:1")

    if num_thread > 1:
        NetworkConfig.NUM_THREADS = num_thread
        print('Set NUM_THREADS = ', num_thread)

    train_insts = TreeReader.read_insts(train_file, True, num_train)
    dev_insts = TreeReader.read_insts(dev_file, False, num_dev)
    test_insts = TreeReader.read_insts(test_file, False, num_test)


    vocab2id = {START:0, STOP:1, UNK:2}
    tag2id = {START:0, STOP:1, UNK:2}
    label2id = {():0}

    for inst in train_insts:  #+ dev_insts + test_insts:
        for word, tag in inst.input:
            if word not in vocab2id:
                vocab2id[word] = len(vocab2id)
            if tag not in tag2id:
                tag2id[tag] = len(tag2id)

    for inst in train_insts: # + dev_insts + test_insts:
        inst.word_seq = torch.tensor([vocab2id[word] for word, tag in [(START, START)] + inst.input + [(STOP, STOP)]]).to(NetworkConfig.DEVICE)
        inst.tag_seq = torch.tensor([tag2id[tag] for word, tag in [(START, START)] + inst.input + [(STOP, STOP)]]).to(NetworkConfig.DEVICE)
        inst.output = inst.output.convert()

        nodes = [inst.output]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                if node.label not in label2id:
                    label2id[node.label] = len(label2id)
                nodes.extend(reversed(node.children))
            else:
                pass

    print('label2id:',list(label2id.keys()))


    for inst in dev_insts + test_insts:
        inst.word_seq = torch.tensor([vocab2id[word] if word in vocab2id else vocab2id[UNK] for word, tag in [(START, START)] + inst.input + [(STOP, STOP)]]).to(NetworkConfig.DEVICE)
        inst.tag_seq = torch.tensor([tag2id[tag] if word in vocab2id else tag2id[UNK] for word, tag in [(START, START)] + inst.input + [(STOP, STOP)]]).to(NetworkConfig.DEVICE)


    gnp = TensorGlobalNetworkParam(len(label2id))
    gnp.ignore_transition = True

    fm = TreeNeuralBuilder(gnp, len(vocab2id), 100, len(tag2id), 50)
    fm.load_pretrain(vocab2id)

    compiler = TreeNetworkCompiler(label2id)

    evaluator = constituent_eval()

    model = NetworkModel(fm, compiler, evaluator)
    model.model_path = "best_parsingtree.pt"

    if batch_size == 1:
        model.learn(train_insts, num_iter, dev_insts)
    else:
        model.learn_batch(train_insts, num_iter, dev_insts, batch_size)

    #model.load_state_dict(torch.load(model.model_path))
    model.load()

    results = model.test(test_insts)
    for inst in results:
        print(inst.get_input())
        print(inst.get_output())
        print(inst.get_prediction())
        print()

    ret = model.evaluator.eval(test_insts)
    print(ret)


