import torch
import torch.nn as nn


class GlobalNetworkParam(nn.Module):

    def __init__(self, label_size):
        super(GlobalNetworkParam, self).__init__()
        self.locked = False
        self._size = 0
        self.label_size =label_size

        self.tuple2id = {}
        self.transition_mat = None


    def is_locked(self):
        return self.locked


    def size(self):
        return self._size


    def transition(self, parent_label_id, tuple_id):
        # tuple_id = self.tuple2id[children_label_ids]
        return self.transition_mat[parent_label_id][tuple_id]


    def finalize_transition(self):
        self.tuple_size = len(self.tuple2id)

        self.transition_mat = []
        for i in range(self.label_size):
            x = []
            for j in range(self.tuple_size):
                x.append(nn.Parameter(torch.tensor(0.0)))
            self.transition_mat.append(x)

        # self.transition_mat = nn.Parameter(torch.zeros(self.label_size, self.tuple_size))


    def add_transition(self, transition):
        parent_label_id, children_label_ids = transition

        if children_label_ids not in self.tuple2id:
            tuple2id_size = len(self.tuple2id)
            self.tuple2id[children_label_ids] = tuple2id_size
        return self.tuple2id[children_label_ids]
        # self.transition_mat[parent_label_id][self.tuple2id[children_label_ids]] = 0







