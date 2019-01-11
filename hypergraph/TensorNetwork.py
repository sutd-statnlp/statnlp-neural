from abc import abstractmethod
# import NetworIDMapper
from hypergraph.NetworkIDMapper import *
import math
import numpy as np
from hypergraph.Utils import *



class TensorNetwork:

    def __init__(self, network_id, instance, fm, num_stage = -1, num_row = -1, num_hyperedge = -1):
        self.network_id = network_id
        self.inst = instance
        self.fm = fm
        self.gnp = fm.gnp
        self.nodeid2labelid = {}
        self.node2hyperedge = []

        self.num_stage = num_stage
        self.num_row = num_row
        self.num_hyperedge = num_hyperedge

        self.size = self.num_stage  * self.num_row
        self.neg_inf_idx = -1
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


        #emissions = torch.tensor([self.fm.extract_helper(self, k) if self.get_node(k) > -1 else -float('inf') for k in range(self.size)])

        # emissions = torch.Tensor(self.size).fill_(-math.inf) #[-math.inf] * self.size #
        #
        # for k in range(self.size):
        #     if self.get_node(k) > -1:
        #         emissions[k] = self.fm.extract_helper(self, k)

        emissions = [self.fm.get_nn_score(self, k) if self.get_node(k) > -1 else torch.tensor(-float('inf')).to(NetworkConfig.DEVICE) for k in range(self.size)]
        emissions = torch.stack(emissions, 0)


        # emissions = torch.tensor([0.0 for k in range(self.size)])
        emissions = emissions.view(self.num_stage, self.num_row)



        for stage_idx in range(self.num_stage):
            if stage_idx == 0:
                score = emissions[stage_idx] #.expand(self.num_row, self.num_hyperedge)

                self.inside_scores[stage_idx] = score

            else:
                childrens_stage = self.get_children(stage_idx)
                #max_number, max_num_hyperedges, _ = childrens_stage.shape #max_number, max_num_hyperedges, 2

                # inside_view = self.inside_scores.view(1, 1, -1).expand(self.num_row, self.num_hyperedge, -1)
                # for_expr = torch.sum(torch.gather(inside_view, 2, childrens_stage), 2)  # max_number, max_num_hyperedges
                for_expr = torch.sum(torch.take(self.inside_scores, childrens_stage), 2)  # this line is same as the above two lines

                # transition_view = self.gnp.transition_mat.view(1,-1).expand(self.num_row, -1)
                # trans_expr = torch.gather(transition_view, 1, self.trans_id[stage_idx])
                trans_expr = torch.take(self.gnp.transition_mat, self.trans_id[stage_idx])  #this line is same as the above two lines


                #trans_expr[trans_expr==-float("inf")] = -float("inf")

                score =for_expr +  trans_expr + emissions[stage_idx].view(self.num_row, 1).expand(self.num_row, self.num_hyperedge) # +
                self.inside_scores[stage_idx] = logSumExp(score) #torch.max(score, 1) #


        final_inside = self.get_insides()

        if math.isinf(final_inside) and final_inside > 0:
            raise Exception("Error: network (ID=", self.network_id, ") has zero inside score")

        weight = self.get_instance().weight
        return final_inside * weight

    def get_insides(self):
        #print('self.inside_scores[-2]:',self.inside_scores[-2])
        return self.inside_scores[-2][0]


    def get_label_id(self, node_k):
        if node_k not in self.nodeid2labelid:
            self.nodeid2labelid[node_k] = self.fm.get_label_id(self, node_k)

        return self.nodeid2labelid[node_k]

    def touch(self):

        self.trans_id = np.full((self.num_stage, self.num_row, self.num_hyperedge), 0) #num_stage, max_number, max_num_hyperedges

        for stage_idx in range(1, self.num_stage):
            self.touch_stage(stage_idx)

        ## long() will be used in torch version  >= 0.4.1
        self.children = torch.from_numpy(self.children).to(NetworkConfig.DEVICE)
        self.trans_id = torch.from_numpy(self.trans_id).to(NetworkConfig.DEVICE)
        if torch.__version__ == "0.4.1":
            self.children = self.children.long()
            self.trans_id = self.trans_id.long()

    def touch_stage(self, stage_idx):

        children_list_k = self.get_children(stage_idx)
        #children_list_k = children_list_k.data.numpy()
        #max_number, max_num_hyperedges, _ = childrens_stage.shape #max_number, max_num_hyperedges, 2

        for idx in range(len(children_list_k)):
            node_id_new = idx + (stage_idx * self.num_row)

            if self.get_node(node_id_new) > -1:
                parent_label_id = self.get_label_id(node_id_new)

                for children_k_index in range(len(children_list_k[idx])):
                    children_k = children_list_k[idx][children_k_index]
                    rhs = [self.get_label_id(child_k) for child_k in children_k if child_k < self.size - 2]
                    if len(rhs) > 0:
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
        self._max = torch.Tensor(self.num_stage + 1, self.num_row).fill_(-math.inf)  # self.getMaxSharedArray()
        self._max[-1][self.zero_idx] = 0
        self._max = self._max.to(NetworkConfig.DEVICE)
        self._max_paths = torch.IntTensor(self.num_stage, self.num_row,2).fill_(-1)  # self.getMaxPathSharedArray()
        self._max_paths = self._max_paths.to(NetworkConfig.DEVICE)

        #emissions = torch.tensor([self.fm.extract_helper(self, k) if self.get_node(k) > -1 else 0 for k in range(self.size)])
        emissions = [self.fm.get_nn_score(self, k) if self.get_node(k) > -1 else torch.tensor(-float('inf')).to(NetworkConfig.DEVICE) for k in range(self.size)]
        emissions = torch.stack(emissions, 0)
        emissions = emissions.view(self.num_stage, self.num_row)

        for stage_idx in range(self.num_stage):
            if stage_idx == 0:
                score = emissions[stage_idx].view(-1, 1).expand(self.num_row, self.num_hyperedge)
            else:
                childrens_stage = self.get_children(stage_idx)
                # max_number, max_num_hyperedges, _ = childrens_stage.shape  # max_number, max_num_hyperedges, 2

                inside_view = self._max.view(1, 1, -1).expand(self.num_row, self.num_hyperedge, -1)
                for_expr = torch.sum(torch.gather(inside_view, 2, childrens_stage), 2)  # max_number, max_num_hyperedges

                transition_view = self.gnp.transition_mat.view(1, -1).expand(self.num_row, -1)
                trans_expr = torch.gather(transition_view, 1, self.trans_id[stage_idx])

                score = for_expr + trans_expr + emissions[stage_idx].view(self.num_row, 1).expand(self.num_row, self.num_hyperedge)  ## max_number, max_number_hyperedge

            self._max[stage_idx], max_id_list = torch.max(score, 1)  # max_id_list: max_number

            if stage_idx > 0:
                max_id_list = max_id_list.view(self.num_row, 1, 1).expand(self.num_row, 1, 2)
                self._max_paths[stage_idx] = torch.gather(childrens_stage, 1, max_id_list).squeeze(1) ## max_number, 2

        ## self.max_path   num_stage x max_number x 2
        ### 1d [num_stage x max_number] [2]
        self._max_paths = self._max_paths.view(-1, 2).cpu().numpy()

        self.non_exist_node_id = (self.num_stage + 1) * self.num_row - 1 -1


    def get_max_path(self, k):
        ## TODO: children might contains non exist node ids careful
        return self._max_paths[k]

