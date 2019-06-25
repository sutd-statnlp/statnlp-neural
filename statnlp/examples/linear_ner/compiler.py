from statnlp.hypergraph.NetworkCompiler import NetworkCompiler
from statnlp.hypergraph.NetworkIDMapper import NetworkIDMapper
from statnlp.hypergraph.TensorBaseNetwork import TensorBaseNetwork
import numpy as np
from typing import Dict

class NERCompiler(NetworkCompiler):

    def __init__(self, label_map: Dict, max_size:int = 20):
        super().__init__()
        self.labels = [None] * len(label_map)
        self.label2id = label_map
        for key in self.label2id:
            self.labels[self.label2id[key]] = key

        NetworkIDMapper.set_capacity(np.asarray([max_size, len(self.labels), 3], dtype=np.int64))

        print("Building generic network..")
        self.build_generic_network(max_size)

    def to_root(self, size):
        return self.to_node(size - 1, len(self.labels) - 1, 2)

    def to_tag(self, pos, label_id):
        return self.to_node(pos, label_id, 1)

    def to_leaf(self, ):
        return self.to_node(0, 0, 0)

    def to_node(self, pos, label_id, node_type):
        return NetworkIDMapper.to_hybrid_node_ID(np.asarray([pos, label_id, node_type]))

    def compile_labeled(self, network_id, inst, param):

        builder = TensorBaseNetwork.NetworkBuilder.builder()
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
        builder = TensorBaseNetwork.NetworkBuilder.builder()
        root_node = self.to_root(inst.size())
        all_nodes = self._all_nodes
        root_idx = np.argwhere(all_nodes == root_node)[0][0]
        node_count = root_idx + 1
        network = builder.build_from_generic(network_id, inst, self._all_nodes, self._all_children, node_count, self.num_hyperedge, param, self)
        return network


    def build_generic_network(self, max_size):
        builder = TensorBaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        builder.add_node(leaf)
        children = [leaf]
        for i in range(max_size):
            current = [None for k in range(len(self.labels))]
            for l in range(len(self.labels)):
                tag_node = self.to_tag(i, l)
                builder.add_node(tag_node)
                for child in children:
                    builder.add_edge(tag_node, [child])
                current[l] = tag_node
            children = current
            root = self.to_root(i+1)
            builder.add_node(root)
            for child in children:
                builder.add_edge(root, [child])
        self._all_nodes, self._all_children, self.num_hyperedge = builder.pre_build()

    def decompile(self, network):
        inst = network.get_instance()
        size = inst.size()
        root_node = self.to_root(size)
        all_nodes = network.get_all_nodes()
        curr_idx = np.argwhere(all_nodes == root_node)[0][0] #network.count_nodes() - 1 #self._all_nodes.index(root_node)
        prediction = [None for i in range(size)]
        for i in range(size):
            children = network.get_max_path(curr_idx)
            child = children[0]
            child_arr = network.get_node_array(child)
            prediction[size - i - 1] = self.labels[child_arr[1]]
            curr_idx = child
        inst.set_prediction(prediction)
        return inst

