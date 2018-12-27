from hypergraph.NetworkCompiler import NetworkCompiler
import numpy as np
from hypergraph.NetworkIDMapper import NetworkIDMapper
from hypergraph.BaseNetwork import BaseNetwork
from hypergraph.GlobalNetworkParam import GlobalNetworkParam
from hypergraph.FeatureManager import FeatureManager
from hypergraph.NetworkModel import NetworkModel
import torch.nn as nn
from hypergraph.Utils import *
from common.LinearInstance import LinearInstance
from example.eval import nereval

class TagNetworkCompiler(NetworkCompiler):

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.label2id = {}
        print(self.labels)
        for i in range(len(self.labels)):
            # print(self.labels[i])
            self.label2id[labels[i]] = i

        NetworkIDMapper.set_capacity(np.asarray([200, 100, 3], dtype=np.int64))

        print(self.label2id)
        print(self.labels)
        self._all_nodes = None
        self._all_children = None
        self._max_size = 100

        # self.build_generic_network()

    def to_root(self, size):
        return self.to_node(size - 1, len(self.labels) - 1, 2)

    def to_tag(self, pos, label_id):
        return self.to_node(pos, label_id, 1)

    def to_leaf(self, ):
        return self.to_node(0, 0, 0)

    def to_node(self, pos, label_id, node_type):
        return NetworkIDMapper.to_hybrid_node_ID(np.asarray([pos, label_id, node_type]))

    def compile_labeled(self, network_id, inst, param):
        builder = BaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        builder.add_node(leaf)

        output = inst.get_output()
        children = [leaf]
        for i in range(inst.size()):
            label = output[i]
            tag_node = self.to_tag(i, self.label2id[label])
            builder.add_node(tag_node)
            builder.add_edge(tag_node, children)
            children = [tag_node]
        root = self.to_root(inst.size())
        builder.add_node(root)
        builder.add_edge(root, children)
        network = builder.build(network_id, inst, param, self)
        return network

    def compile_unlabeled(self, network_id, inst, param):

        builder = BaseNetwork.NetworkBuilder.builder()
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

    # def build_generic_network(self, ):
    #
    #     network = builder.build(None, None, None, None)
    #     self._all_nodes = network.get_all_nodes()
    #     self._all_children = network.get_all_children()

    def decompile(self, network):
        inst = network.get_instance()

        size = inst.size()
        root_node = self.to_root(size)
        curr_idx = network.count_nodes() - 1 #self._all_nodes.index(root_node)
        prediction = [None for i in range(size)]
        for i in range(size):
            children = network.get_max_path(curr_idx)
            child = children[0]
            child_arr = network.get_node_array(child)
            prediction[size - i - 1] = self.labels[child_arr[1]]
            curr_idx = child

        inst.set_prediction(prediction)
        return inst


class TagFeatureManager(FeatureManager):
    def __init__(self, param_g, voc_size):
        super().__init__(param_g)
        self.token_embed = 100
        self.word_embed = nn.Embedding(voc_size, self.token_embed)
        self.rnn = nn.LSTM(self.token_embed, self.token_embed, batch_first=True,bidirectional=True)

        self.linear = nn.Linear(self.token_embed * 2, param_g.label_size)


    def load_pretrain(self, path, word2idx):
        emb = load_emb_glove(path, word2idx, self.token_embed)
        self.word_embed.from_pretrained(torch.FloatTensor(emb), freeze=False)

    # @abstractmethod
    # def extract_helper(self, network, parent_k, children_k, children_k_index):
    #     pass
    def build_nn_graph(self, instance):

        word_vec = self.word_embed(instance.word_seq).unsqueeze(0)

        lstm_out, _ = self.rnn(word_vec, None)
        linear_output = self.linear(lstm_out).squeeze(0)
        return linear_output


    def extract_helper(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)  # pos, label_id, node_type
        pos = parent_arr[0]
        label_id = parent_arr[1]
        node_type = parent_arr[2]

        if node_type == 0 or node_type == 2: #Start, End
            return torch.tensor(0.0)
        else:
            nn_output = network.nn_output
            return nn_output[pos][label_id]


    def get_label_id(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        return parent_arr[1]


class TagReader():
    label2id_map = {}
    label2id_map["<START>"] = 0
    @staticmethod
    def read_insts(file, is_labeled, number):
        insts = []
        inputs = []
        outputs = []
        f = open(file, 'r', encoding='utf-8')
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

                if len(insts) >= number and number > 0:
                    break

            else:
                fields = line.split()
                input = fields[0]
                output = fields[-1]

                # if output.endswith("NP"):
                #     output = "NP"
                # else:
                #     output = "O"

                if not output in TagReader.label2id_map:
                    output_id = len(TagReader.label2id_map)
                    TagReader.label2id_map[output] = output_id
                else:
                    output_id = TagReader.label2id_map[output]

                inputs.append(input)
                outputs.append(output)

        f.close()

        return insts


if __name__ == "__main__":

    torch.manual_seed(9997)


    train_file = "data/conll/train.txt.bieos"
    dev_file = "data/conll/dev.txt.bieos"
    test_file = "data/conll/test.txt.bieos"
    trial_file = "data/conll/trial.txt.bieos"

    # train_file = train_file
    # dev_file = train_file
    # test_file = train_file

    data_size = -1
    num_iter = 100

    train_insts = TagReader.read_insts(train_file, True, data_size)
    dev_insts = TagReader.read_insts(dev_file, False, data_size)
    test_insts = TagReader.read_insts(test_file, False, data_size)
    TagReader.label2id_map["<ROOT>"] = len(TagReader.label2id_map)

    vocab2id = {}
    for inst in train_insts + dev_insts + test_insts:
        for word in inst.input:
            if word not in vocab2id:
                vocab2id[word] = len(vocab2id)

    for inst in train_insts + dev_insts + test_insts:
        inst.word_seq = torch.tensor([vocab2id[word] for word in inst.input])



    gnp = GlobalNetworkParam(len(TagReader.label2id_map))
    fm = TagFeatureManager(gnp, len(vocab2id))
    #fm.load_pretrain('data/glove.6B.100d.txt', vocab2id)
    print(list(TagReader.label2id_map.keys()))
    compiler = TagNetworkCompiler(list(TagReader.label2id_map.keys()))


    evaluator = nereval()
    print('Start Training...', flush=True)
    model = NetworkModel(fm, compiler, evaluator)
    model.learn(train_insts, num_iter, dev_insts)

    results = model.test(test_insts)

    print()
    print('Result:')
    corr = 0
    total = 0
    for i in range(len(test_insts)):
        inst = results[i]
        total += inst.size()
        for pos in range(inst.size()):
            if inst.get_output()[pos] == inst.get_prediction()[pos]:
                corr += 1
        print("resulit is :", results[i].get_prediction())

    print("accuracy: ", str(corr*1.0 / total))

