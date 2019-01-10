from hypergraph.NetworkIDMapper import NetworkIDMapper
from hypergraph.Utils import *
import numpy as np
from hypergraph.BaseNetwork import BaseNetwork
from hypergraph.TensorTableLookupNetwork import TensorTableLookupNetwork


class TensorBaseNetwork(TensorTableLookupNetwork):
    def __init__(self, network_id, instance, nodes, children, node_count, param, compiler, num_stage, num_row, num_hyperedge):
        super().__init__(network_id, instance, nodes, children, node_count, param, compiler, num_stage, num_row, num_hyperedge)
        self.node_count = node_count

        self.is_visible = [False for i in range(node_count)]


    def count_nodes(self):
        return self.node_count




    class NetworkBuilder:

        def __init__(self):
            self._children_tmp = {}

        @staticmethod
        def builder():
            return TensorBaseNetwork.NetworkBuilder()

        @staticmethod
        def quick_build(network_id, instance, nodes, children, node_count, param, compiler, num_stage, num_row, num_hyperedge):
            return TensorBaseNetwork(network_id, instance, nodes, children, node_count, param, compiler, num_stage, num_row, num_hyperedge)

        def add_node(self, node):
            if node in self._children_tmp:
                return False
            else:
                self._children_tmp[node] = None
                return True

        def add_edge(self, parent, child):
            self.check_link_validity(parent, child)
            if parent not in self._children_tmp or self._children_tmp[parent] == None:
                self._children_tmp[parent] = []
            existing_children = self._children_tmp[parent]
            for k in range(len(existing_children)):
                if existing_children[k] == child:
                    return False
            existing_children.append(child)
            return True

        def num_nodes_tmp(self):
            return len(self._children_tmp)

        def get_children_tmp(self, node):
            return self._children_tmp[node]

        def get_nodes_tmp(self):
            nodes = []
            for key in self._children_tmp:
                nodes.append(key)
            return nodes

        def remove_tmp(self, node):
            if node not in self._children_tmp:
                return False
            self._children_tmp.pop(node)
            return True

        def contains_node(self, node):
            return node in self._children_tmp

        def contains_edge(self, parent, child):
            if parent not in self._children_tmp:
                return False
            children = self._children_tmp[parent]
            for presentChild in children:
                if presentChild == child:
                    return True
            return False

        def build(self, network_id, instance, param, compiler):

            values = []

            for node in self._children_tmp:
                values.append(node)

            node_list = [0 for i in range(len(self._children_tmp))]
            is_visible = [False for i in range(len(node_list))]

            nodes_value2id_map = {}

            num_hyperedge = -1
            values.sort()
            for k in range(len(values)):
                node_list[k] = values[k]
                is_visible[k] = True
                nodes_value2id_map[node_list[k]] = k
            # node_list.sort()
            children_list = [None for i in range(len(node_list))]

            for parent in self._children_tmp:
                #print("builder parent: ", parent, " chidren_tmp: " , self._children_tmp[parent])
                parent_index = nodes_value2id_map[parent]
                childrens = self._children_tmp[parent]
                if childrens == None:
                    children_list[parent_index] = [[]]  # new int[1][0]

                else:
                    children_list[parent_index] = [None for i in range(len(childrens))]
                    num_hyperedge = max(num_hyperedge, len(children_list[parent_index]))

                    for k in range(len(children_list[parent_index])):
                        children = childrens[k]
                        children_index = []

                        for m in range(len(children)):
                            if children[m] < 0:
                                children_index.append(children[m])
                            else:
                                children_index.append(nodes_value2id_map[children[m]])

                        children_list[parent_index][k] = children_index
                    #print("parent is :", parent, " children_list, ", children_list[parent_index])
            for k in range(len(children_list)):
                if children_list[k] == None:
                    children_list[k] = [[]]

            result = None

            # if network_id != None or instance != None or param != None or compiler != None:
            result = BaseNetwork.NetworkBuilder.quick_build(network_id, instance, node_list, children_list, len(node_list), param, compiler)
            # TODO: handle the case when network_id != None or instance != None or param != None or compiler != None
            # this is for rudimentary network builder

            sorted_nodes, num_row = topological_sort(result)

            num_row = max(num_row, 2)

            num_stage = len(sorted_nodes.keys())
            # inside_stages = np.full((max_number, num_stage), -np.inf)
            #
            # assert max_number >= 3

            # inside_stages[-1][zero_idx] = 0
            # inside_stages[-1][neg_inf_idx] = -inf

            ## node id - > new node id
            ### node array , children_array

            ##
            mapping = {}
            mapping_rev = {}
            idx_stage = 0
            for stage_idx in sorted_nodes:
                for node_k in sorted_nodes[stage_idx]:
                    mapping[node_k] = idx_stage  ## old -> new
                    mapping_rev[idx_stage] = node_k
                    idx_stage += 1

                idx_stage += num_row - len(sorted_nodes[stage_idx])

        ##

            staged_nodes = np.full((num_row * num_stage), -1, dtype=np.long)  ##size = max_number * num_stage
            for i in range(num_row * num_stage):
                if i in mapping_rev:
                    orig_node_k = mapping_rev[i]
                    staged_nodes[i] = values[orig_node_k]

            all_children_list = np.empty((num_stage, num_row, num_hyperedge, 2), dtype=np.long)
            size = num_stage * num_row
            neg_inf_idx = (num_stage + 1) * num_row - 1
            zero_idx = neg_inf_idx - 1
            all_children_list[:, :, :, 0].fill(neg_inf_idx)
            all_children_list[:, :, :, 1].fill(zero_idx)

            old_children_list = children_list

            for stage_idx in range(num_stage):
                col = sorted_nodes[stage_idx]
                for node_id in col:
                    now_node_k = mapping[node_id]

                    curr_row_idx = now_node_k % num_row

                    #all_children_list[stage_idx][now_node_k]
                    for hyperedge_index in range(len(old_children_list[node_id])):
                        hyperedge = old_children_list[node_id][hyperedge_index]
                        hyperedge = [mapping[node_id] for node_id in hyperedge]
                        if len(hyperedge) > 0:
                            all_children_list[stage_idx][curr_row_idx][hyperedge_index][0] = hyperedge[0]
                            if len(hyperedge) > 1:
                                all_children_list[stage_idx][curr_row_idx][hyperedge_index][1] = hyperedge[1]

            result = TensorBaseNetwork.NetworkBuilder.quick_build(network_id, instance, staged_nodes, all_children_list,
                                                                      len(staged_nodes), param, compiler, num_stage, num_row, num_hyperedge)


            result.is_visible = is_visible
            return result


        def check_link_validity(self, parent, children):
            for child in children:
                if child < 0:
                    continue

                if child >= parent:
                    eprint(NetworkIDMapper.to_hybrid_node_array(parent))
                    eprint(NetworkIDMapper.to_hybrid_node_array(child))
                    eprint()
                    raise Exception(
                        "In an edge, the parent needs to have larger node ID in order to have a proper schedule for inference. Violation: ",
                        parent, "\t", children)

            self.check_node_validity(parent)

            for child in children:
                if child < 0:
                    continue

                self.check_node_validity(child)


        def check_node_validity(self, node):
            if node not in self._children_tmp:
                raise Exception("Node not found:", NetworkIDMapper.to_hybrid_node_array(node))






