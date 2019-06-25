from hypergraph.NetworkCompiler import NetworkCompiler
from hypergraph.NetworkIDMapper import NetworkIDMapper
from hypergraph.TensorBaseNetwork import TensorBaseNetwork
import numpy as np
from typing import Dict

class SemiCompiler(NetworkCompiler):

    def __init__(self, label_map, max_size=2, max_seg_size=5):
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
        # self.debug = True
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
            # print(span_tuple)
            left = span_tuple[0]
            right = span_tuple[1]
            label = span_tuple[2]
            tag_node = self.to_tag(left, right, self.label2id[label])
            tag_prime = self.to_tag_prime(right, self.label2id[label+"_prime"])
            builder.add_node(tag_node)
            builder.add_node(tag_prime)
            builder.add_edge(tag_node, children)
            builder.add_edge(tag_prime, [tag_node])
            children = [tag_prime]
        root = self.to_root(inst.size())
        builder.add_node(root)
        builder.add_edge(root, children)
        network = builder.build(network_id, inst, param, self)

        # if self.debug:
        #     unlabel = self.compile_unlabeled(network_id, inst, param)
        #     if not unlabel.contains(network):
        #         print("not contain")


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
        leaf = self.to_leaf()
        builder.add_node(leaf)

        for i in range(inst.size()):
            for j in range(i, min(i + self._max_seg_size, inst.size())):
                for l in range(len(self.labels)):
                    if l == 0 or l == len(self.labels) - 1:
                        continue
                    if i != j and l == self.label2id["O"]:
                        continue
                    if self.labels[l].endswith("_prime"):
                        continue
                    span_node = self.to_tag(i, j, l)
                    added = False
                    for prev_label_id in range(len(self.labels)):
                        if i > 0:
                            if not self.labels[prev_label_id].endswith("_prime"):
                                continue
                            child_node = self.to_tag_prime(i - 1, prev_label_id)
                            if builder.contains_node(child_node):
                                added = True
                                builder.add_node(span_node)
                                builder.add_edge(span_node, [child_node])
                        else:
                            added = True
                            builder.add_node(span_node)
                            builder.add_edge(span_node, [leaf])
                    if added:
                        end_prime_node = self.to_tag_prime(j, self.label2id[self.labels[l] + "_prime"])
                        # end_prime_node = self.to_tag_prime(j, l)
                        builder.add_node(end_prime_node)
                        builder.add_edge(end_prime_node, [span_node])
                        if j == inst.size() - 1:
                            root = self.to_root(j + 1)
                            builder.add_node(root)
                            builder.add_edge(root, [end_prime_node])
        network = builder.build(network_id, inst, param, self)
        return network
        # return self.compile_labeled(network_id, inst, param)
        # builder = TensorBaseNetwork.NetworkBuilder.builder()
        # root_node = self.to_root(inst.size())
        # all_nodes = self._all_nodes
        # root_idx = np.argwhere(all_nodes == root_node)[0][0]
        # node_count = root_idx + 1
        # network = builder.build_from_generic(network_id, inst, self._all_nodes, self._all_children, node_count, self.num_hyperedge, param, self)
        # return network

    def build_generic_network(self, ):

        builder = TensorBaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        builder.add_node(leaf)

        for i in range(self._max_size):
            for j in range(i, min(i + self._max_seg_size, self._max_size)):
                for l in range(len(self.labels)):
                    if l == 0 or l == len(self.labels) - 1:
                        continue
                    if i!= j and l == self.label2id["O"]:
                        continue
                    if self.labels[l].endswith("_prime"):
                        continue
                    span_node = self.to_tag(i, j, l)
                    added = False
                    for prev_label_id in range(len(self.labels)):
                        if i > 0:
                            if not self.labels[prev_label_id].endswith("_prime"):
                                continue
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
                        end_prime_node = self.to_tag_prime(j, self.label2id[self.labels[l] + "_prime"])
                        # end_prime_node = self.to_tag_prime(j, l)
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
            if type == 1: ## label node
                end = child_arr[0]
                start =  end - child_arr[1] + 1
                label = self.labels[child_arr[2]]
                prediction.append((start, end, label))
                # prediction[size - i - 1] = self.labels[child_arr[1]]

            curr_idx = child
        # print(prediction)
        inst.set_prediction(prediction[::-1])
        return inst
