import torch
import torch.nn as nn
from hypergraph.NetworkConfig import NetworkConfig

class TensorGlobalNetworkParam(nn.Module):

    def __init__(self, label_size):
        super(TensorGlobalNetworkParam, self).__init__()
        self.locked = False
        self._size = 0
        self.label_size =label_size

        self.tuple2id = {}
        self.tuple2id[()] = 0
        self.transition_mat = None


    def is_locked(self):
        return self.locked


    def size(self):
        return self._size



    def finalize_transition(self):
        self.tuple_size = len(self.tuple2id)
        self.transition_mat = nn.Parameter(torch.randn(self.tuple_size)).to(NetworkConfig.DEVICE)
        self.transition_mat.data[0] = -float('inf') # padding
        self.locked = True
        print('self.tuple2id:', self.tuple2id)


    def add_transition(self, transition):

        parent_label_id, children_label_ids = transition
        t = tuple([parent_label_id] + children_label_ids)
        if not self.locked and t not in self.tuple2id:
            tuple2id_size = len(self.tuple2id)
            self.tuple2id[t] = tuple2id_size

        return self.tuple2id[t]








