import torch
import torch.nn as nn
from hypergraph.NetworkConfig import NetworkConfig
from termcolor import colored

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
        #print('self.tuple2id:', self.tuple2id)

        # def hookBFunc(m, gi, go):  # 该函数必须是function(grad)这种形式，grad的参数默认给出
        #     print(colored('Bhook:', 'green'))
        #     print(gi, go)
        # #
        # # self.register_backward_hook(hookFunc)
        #
        # def hookFunc(g):
        #     print(colored('hook:', 'red'))
        #     #g[0] = 0
        #     print(g)
        #
        # self.transition_mat.register_hook(hookFunc)
        # self.register_backward_hook(hookBFunc)

    def add_transition(self, transition):

        parent_label_id, children_label_ids = transition
        t = tuple([parent_label_id] + children_label_ids)
        if not self.locked and t not in self.tuple2id:
            tuple2id_size = len(self.tuple2id)
            self.tuple2id[t] = tuple2id_size

        return self.tuple2id[t]








