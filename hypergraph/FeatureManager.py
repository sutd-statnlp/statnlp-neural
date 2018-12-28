from abc import ABC, abstractmethod
import torch.nn as nn

class FeatureManager(nn.Module):

    def __init__(self, param_g):
        super(FeatureManager, self).__init__()
        self._param_g = param_g

    ##initilaize neural network

    def get_param_g(self):
        return self._param_g

    @abstractmethod
    def build_nn_graph(self, instance):
        pass

    @abstractmethod
    def generate_batches(self, train_insts, batch_size):
        pass

    @abstractmethod
    def build_nn_graph_batch(self, batch_input_seqs):
        pass

    @abstractmethod
    def extract_helper(self, network, parent_k):
        ## given a node parent_k, return score.
        # parent_k -> network.nn_output  score
        pass

    def extract(self, network, parent_k):
        score = self.extract_helper(network, parent_k)

        return score
