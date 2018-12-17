from abc import ABC, abstractmethod
# import NetworIDMapper
from NetworkIDMapper import *
import torch
import math
import numpy as np
import torch.autograd as autograd
from Utils import *


class Network:

    def __init__(self, network_id, instance, fm):
        self.network_id = network_id
        self.inst = instance
        self.fm = fm
        self.gnp = fm._param_g
        self.nodeid2labelid = {}
        self.node2hyperedge = {}

    def get_network_id(self):
        return self.network_id

    ##TODO: multithread
    def get_thread_id(self):
        pass

    def get_instance(self):
        return self.inst



    def inside(self):
        self.inside_scores = [torch.tensor(0)] * self.count_nodes()
        for k in range(self.count_nodes()):
            self.get_inside(k)
        if math.isinf(self.get_insides()) and self.get_insides() > 0:
            raise Exception("Error: network (ID=", self.network_id, ") has zero inside score")


        weight = self.get_instance().weight
        return self.get_insides() * weight

    def get_insides(self):
        return self.inside_scores[self.count_nodes() - 1]

    def get_inside(self, k):
        if self.is_removed(k):
            self.inside_scores[k] = torch.tensor(-math.inf)
            return

        current_label_id = self.get_label_id(k)

        # inside_score = torch.tensor(-math.inf)
        children_list_k = self.get_children(k)
        ## If this node has no child edge, assume there is one edge with no child node
        ## This is done so that every node is visited in the feature extraction step below

        if len(children_list_k) > 0:

            size = len(children_list_k)
            # for_expr = torch.zeros(size)
            trans = torch.zeros(size)
            ## parent_k -> children_k_index -> tuple_id
            # for_list = []
            # trans_list = []
            emission = self.fm.extract_helper(self, k)
            # emission = emission.expand(size)
            score_list = []
            for children_k_index in range(len(children_list_k)):
                children_k = children_list_k[children_k_index]
                # for_expr[children_k_index] = sum([self.inside_scores[child_k] for child_k in children_k])
                #torch.tensor(np.sum(np.take(self.inside_scores, children_k), dtype=float))
                # for_list.append(sum([self.inside_scores[child_k] for child_k in children_k]).unsqueeze(0) if len(
                #     children_k) > 0 else torch.tensor([0.0]))
                # trans_list.append(
                #         self.gnp.transition_mat[current_label_id][self.node2hyperedge[k][children_k_index]].unsqueeze(0))
                # trans[children_k_index] = self.gnp.transition(current_label_id, tuple([self.get_label_id(child_k) for child_k in children_k]))
                leng = len(children_k)
                s1 = self.inside_scores[children_k[0]] if leng == 1 else (
                            self.inside_scores[children_k[0]] + self.inside_scores[
                        children_k[1]]) if leng == 2 else torch.tensor(0.0)
                s2 = self.gnp.transition_mat[current_label_id][self.node2hyperedge[k][children_k_index]]
                s3 = emission
                score_list.append((s1 + s2 + s3).unsqueeze(0))
                # trans[children_k_index] = self.gnp.transition_mat[current_label_id][self.node2hyperedge[k][children_k_index]]

            ## emission
            # for_expr = torch.cat(for_list)
            # trans = torch.cat(trans_list)

            # score = for_expr + trans + emission
            score = torch.cat(score_list)
            self.inside_scores[k] = log_sum_exp(score)
        else:  # This is a sink node
            self.inside_scores[k] = torch.tensor(0)


    def get_label_id(self, node_k):
        if node_k not in self.nodeid2labelid:
            self.nodeid2labelid[node_k] = self.fm.get_label_id(self, node_k)

        return self.nodeid2labelid[node_k]


    def touch(self):
        for k in range(self.count_nodes()):
            self.touch_node(k)


    def touch_node(self, k):
        '''
        :param k:
        :return:
        '''
        if self.is_removed(k):
            return

        children_list_k = self.get_children(k)
        parent_label_id = self.get_label_id(k)
        self.node2hyperedge[k] = []
        for children_k_index in range(len(children_list_k)):
            children_k = children_list_k[children_k_index]
            rhs = tuple([self.get_label_id(child_k) for child_k in children_k])
            # self.gnp.add_transition((parent_label_id, rhs))
            tuple_id = self.gnp.add_transition((parent_label_id, rhs))
            self.node2hyperedge[k].append(tuple_id)



    def get_node_array(self, k):
        node = self.get_node(k)
        return NetworkIDMapper.to_hybrid_node_array(node)


    @abstractmethod
    def get_children(self, k) -> np.ndarray:
        pass


    @abstractmethod
    def get_node(self, k):
        pass


    @abstractmethod
    def count_nodes(self) -> int:
        pass


    @abstractmethod
    def is_removed(self, k):
        pass

    def max(self):
        self._max = [torch.tensor(0.0)] * self.count_nodes() # self.getMaxSharedArray()
        self._max_paths = [torch.tensor(0)] * self.count_nodes() # self.getMaxPathSharedArray()
        for k in range(self.count_nodes()):
            self.maxk(k)



    def get_max_path(self, k):

        return self._max_paths[k]


    def maxk(self, k):
        if self.is_removed(k):
            self._max[k] = float("-inf")
            return

        children_list_k = self.get_children(k)
        self._max[k] = float("-inf")

        current_label_id = self.get_label_id(k)
        emission = self.fm.extract_helper(self, k)

        for children_k_index in range(len(children_list_k)):
            children_k = children_list_k[children_k_index]


            #fa = self.param.extract(self, k, children_k, children_k_index)
            #score = fa.get_score(self.param)

            transition = self.gnp.transition(current_label_id, tuple([self.get_label_id(child_k) for child_k in children_k]))
            score = transition + emission

            score += sum([self._max[child_k] for child_k in children_k])

            # for child_k in children_k:
            #     score += self._max[child_k]

            # print('maxk:',type(score), '\t', type(self._max[k]))
            if score >= self._max[k]:
                self._max[k] = score
                self._max_paths[k] = children_k
