from abc import abstractmethod
# import NetworIDMapper
from hypergraph.NetworkIDMapper import *
import math
import numpy as np
from hypergraph.Utils import *



class TensorNetwork:

    def __init__(self, network_id, instance, fm):
        self.network_id = network_id
        self.inst = instance
        self.fm = fm
        self.gnp = fm._param_g
        self.nodeid2labelid = {}
        self.node2hyperedge = []

        self.size = self.num_stage * self.num_row
        self.neg_inf_idx = self.size -1
        self.zero_idx = self.neg_inf_idx - 1


    def get_network_id(self):
        return self.network_id

    ##TODO: multithread
    def get_thread_id(self):
        pass

    def get_instance(self):
        return self.inst

    def inside(self):
        # self.inside_scores = [torch.tensor([-math.inf])] * (self.count_nodes() + 1)  #[torch.tensor([0.0])] * self.count_nodes()
        self.inside_scores = torch.Tensor(self.num_stage + 1, self.num_row).fill_(-math.inf)
        '''
        inside_scores is set to self.num_stage + 1 because we do not want the last column overwritten
        '''
        self.inside_scores[-1][self.zero_idx] = 0
        self.inside_scores = self.inside_scores.to(NetworkConfig.DEVICE)


        emissions = torch.tensor([self.fm.extract_helper(self, k) if self.get_node(k) > -1 else 0 for k in range(self.size)])
        emissions = emissions.view(self.num_stage, self.num_row)



        for stage_idx in range(self.num_stage):
            if stage_idx == 0:
                score = emissions[stage_idx]
            else:
                childrens_stage = self.get_children(stage_idx)
                max_number, max_num_hyperedges, _ = childrens_stage.shape #max_number, max_num_hyperedges, 2

                inside_view = self.inside_scores.view(1, -1).expand(max_number, max_num_hyperedges, -1)
                for_expr = torch.sum(torch.gather(inside_view, 2, childrens_stage), 1)  #max_number, max_num_hyperedges

                transition_view = self.gnp.transition_mat.expand(max_number, -1)
                trans_expr = torch.gather(transition_view, 1, self.trans_id[stage_idx])

                score =for_expr + trans_expr + emissions[stage_idx]

            self.inside_scores[stage_idx] = logSumExp(score)


        final_inside = self.get_insides()

        if math.isinf(final_inside) and final_inside > 0:
            raise Exception("Error: network (ID=", self.network_id, ") has zero inside score")

        weight = self.get_instance().weight
        return final_inside * weight

    def get_insides(self):
        return self.inside_scores[-2][0]


    def get_label_id(self, node_k):
        if node_k not in self.nodeid2labelid:
            self.nodeid2labelid[node_k] = self.fm.get_label_id(self, node_k)

        return self.nodeid2labelid[node_k]

    def touch(self):

        self.empty_idx = self.gnp.tuple_size
        self.trans_id = np.full((self.num_stage, self.max_number, self.max_num_hyperedges), self.empty_idx) #num_stage, max_number, max_num_hyperedges

        for stage_idx in range(self.num_stage):
            self.touch_stage(stage_idx)


    def touch_stage(self, stage_idx):

        children_list_k = self.get_children(stage_idx).data.numpy()
        #max_number, max_num_hyperedges, _ = childrens_stage.shape #max_number, max_num_hyperedges, 2

        for idx in range(len(children_list_k)):
            node_id_new = idx + (stage_idx * self.num_row)

            if self.get_node(node_id_new) > -1:
                parent_label_id = self.get_label_id(node_id_new)

                for children_k_index in range(len(children_list_k[idx])):
                    children_k = children_list_k[idx][children_k_index]
                    rhs = tuple([self.get_label_id(child_k) for child_k in children_k])
                    transition_id = self.gnp.add_transition((parent_label_id, rhs))
                    self.trans_id[stage_idx][idx][children_k_index] = transition_id



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
        self._max = torch.Tensor(self.count_nodes() + 1).fill_(-math.inf)  # self.getMaxSharedArray()
        self._max[-1] = 0.0
        self._max = self._max.to(NetworkConfig.DEVICE)
        self._max_paths = [-1] * self.count_nodes()  # self.getMaxPathSharedArray()
        for k in range(self.count_nodes()):
            self.maxk(k)

    def get_max_path(self, k):

        return self._max_paths[k]

    def maxk(self, k):

        children_list_k = self.get_children(k)
        size = len(children_list_k)
        current_label_id = self.nodeid2labelid[k]
        emission_expr = self.fm.extract_helper(self, k)

        if len(children_list_k[0]) > 0:
            children_list_index_tensor = self.nodeid2childrenids[k]
            max_view = self._max.view(1, self.count_nodes() + 1).expand(size, self.count_nodes() + 1)
            for_expr = torch.sum(torch.gather(max_view, 1, children_list_index_tensor), 1)

            tuple_list_tensor = self.node2hyperedge[k]
            trans_expr = torch.gather(self.gnp.transition_mat[current_label_id], 0, tuple_list_tensor)

            score = for_expr + trans_expr + emission_expr

        else:
            score = emission_expr


        self._max[k], max_id = torch.max(score, 0)
        self._max_paths[k] = children_list_k[max_id]

