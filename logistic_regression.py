from NetworkCompiler import NetworkCompiler
import numpy as np
from NetworkIDMapper import NetworkIDMapper
from BaseNetwork import BaseNetwork
from GlobalNetworkParam import GlobalNetworkParam
from Instance import Instance
from FeatureManager import FeatureManager
from FeatureArray import FeatureArray
from NetworkModel import NetworkModel
from enum import Enum
import torch


class NodeType(Enum):
    LEAF = 0
    NODE = 1
    ROOT = 2


class LRInstance(Instance):
    def __init__(self, instance_id, weight, input, output):
        super().__init__(instance_id, weight, input, output)


    def size(self):
        return len(input)

    def duplicate(self):
        dup = LRInstance(self.instance_id, self.weight, self.input, self.output)
        return dup

    def removeOutput(self):
        self.output = None

    def removePrediction(self):
        self.prediction = None

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, prediction):
        self.prediction = prediction

    def has_output(self):
        return self.output != None

    def has_prediction(self):
        return self.prediction != None


    def __str__(self):
        return 'input:' + str(self.input) + '\toutput:' + str(self.output) + ' is_labeled:' + str(self.is_labeled)


class LRFeatureManager(FeatureManager):
    def __init__(self, param_g):
        super().__init__(param_g)


    # @abstractmethod
    # def extract_helper(self, network, parent_k, children_k, children_k_index):
    #     pass
    def extract_helper(self, network, parent_k, children_k, children_k_index):
        parent_arr = network.get_node_array(parent_k)
        node_type_id = parent_arr[0]
        if node_type_id == 0 or node_type_id == 2:
            return FeatureArray.EMPTY
        inst = network.get_instance()
        #print("input ", inst.get_input(), " labeled: ", inst.is_labeled)
        ft_strs = inst.get_input().split(",")

        fs = []
        label_id = str(parent_arr[1])

        fs.append(self._param_g.to_feature(network, "location", label_id, ft_strs[0]))
        fs.append(self._param_g.to_feature(network, "quality", label_id, ft_strs[1]))
        fs.append(self._param_g.to_feature(network, "people", label_id, ft_strs[2]))
        # print('parent_arr:', parent_arr)
        # print(fs)
        return self.create_feature_array(network, fs)


##Done
class LRReader():

    label2id_map = {}

    @staticmethod
    def read_insts(file, is_labeled, number):
        insts = []
        f = open(file, 'r', encoding='utf-8')
        for line in f:
            line = line.strip()
            fields = line.split(' ')
            inputs = fields[0]
            output = fields[1]

            if not output in LRReader.label2id_map:
                output_id = len(LRReader.label2id_map)
                LRReader.label2id_map[output] = output_id
            else:
                output_id = LRReader.label2id_map[output]

            inst = LRInstance(len(insts) + 1, 1, inputs, output_id)
            if is_labeled:
                inst.set_labeled()
            else:
                inst.set_unlabeled()
            insts.append(inst)
        f.close()

        return insts


class LRNetworkCompiler(NetworkCompiler):

    def __init__(self):
        ##node type and label id
        NetworkIDMapper.set_capacity(np.asarray([3, 2], dtype=np.int64))
        self._all_nodes = None
        self._all_children = None
        self.build_generic_network()

    def to_leaf(self):
        return NetworkIDMapper.to_hybrid_node_ID(np.asarray([0, 0], dtype=np.int64))

    def to_node(self, label_id):
        return NetworkIDMapper.to_hybrid_node_ID(np.asarray([1, label_id], dtype=np.int64))

    def to_root(self):
        return NetworkIDMapper.to_hybrid_node_ID(np.asarray([2, 0], dtype=np.int64))

    def compile_labeled(self, network_id, inst, param):
        builder = BaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        builder.add_node(leaf)

        node = self.to_node(inst.get_output())
        builder.add_node(node)
        builder.add_edge(node, [leaf])

        root = self.to_root()
        builder.add_node(root)
        builder.add_edge(root, [node])

        network = builder.build(network_id, inst, param, self)

        return network

    def compile_unlabeled(self, network_id, inst, param):
        root = self.to_root()
        root_idx = self._all_nodes.index(root)
        num_nodes = root_idx + 1
        #print("all nodes: ", num_nodes)
        network =  BaseNetwork.NetworkBuilder.quick_build(network_id, inst, self._all_nodes, self._all_children, num_nodes,
                                                      param, self)
        # print("inside compile unlabeled: ", network)
        return network

    def build_generic_network(self):
        builder = BaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        leaves = [leaf]
        builder.add_node(leaf)
        root = self.to_root()
        builder.add_node(root)
        for label_id in range(2):
            node = self.to_node(label_id)
            builder.add_node(node)
            builder.add_edge(node, leaves)
            builder.add_edge(root, [node])

        network = builder.build(None, None, None, None)
        self._all_nodes = network.get_all_nodes()
        print("generic all nodes: ", self._all_nodes)
        self._all_children = network.get_all_children()
        print("generic all children: ", self._all_children)


    def decompile(self, network):
        inst = network.get_instance()
        root = self.to_root()
        node_idx = self._all_nodes.index(root)
        label_node_idx = network.get_max_path(node_idx)[0]
        arr = network.get_node_array(label_node_idx)
        label_id = arr[1]
        inst.set_prediction(label_id)
        return inst


if __name__ == "__main__":
    train_file = "train.txt"
    train_insts = LRReader.read_insts(train_file, True, -1)

    torch.manual_seed(1)

    gnp = GlobalNetworkParam()
    fm = LRFeatureManager(gnp)
    compiler = LRNetworkCompiler()

    print('CAPACITY:', NetworkIDMapper.CAPACITY)

    model = NetworkModel(fm, compiler)
    model.train(train_insts, 500)

    test_file = "test.txt"
    test_insts = LRReader.read_insts(train_file, False, -1)
    results = model.test(test_insts)

    print()
    print('Result:')
    for i in range(len(test_insts)):
        print("resulit is :", results[i].get_prediction())

