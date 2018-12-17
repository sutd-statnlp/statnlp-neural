import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from GlobalNetworkParam import GlobalNetworkParam
from NetworkConfig import NetworkConfig
import torch.optim
from Utils import *
import time
from termcolor import colored


class NetworkModel(nn.Module):
    Iter = 0

    def __init__(self, fm, compiler):
        super().__init__()
        self._fm = fm
        self._compiler = compiler
        self._all_instances = None
        self._networks = None

    def get_instances(self):
        return self._all_instances

    def get_feature_manager(self):
        return self._fm

    def get_network_compiler(self):
        return self._compiler

    def split_instances_for_train(self, insts_before_split):
        eprint("#instances=", len(insts_before_split))
        insts = [None for i in range(len(insts_before_split) * 2)]

        k = 0
        for i in range(0, len(insts), 2):
            insts[i] = insts_before_split[k]
            insts[i + 1] = insts_before_split[k].duplicate()
            insts[i + 1].set_instance_id(-insts[i].get_instance_id())
            insts[i + 1].set_weight(-insts[i].get_weight())
            insts[i + 1].set_unlabeled()
            k = k + 1
        return insts

    def lock_it(self):
        gnp = self._fm.get_param_g()

        if gnp.is_locked():
            return

        gnp.finalize_transition()
        gnp.locked = True

    def train(self, train_insts, max_iterations):

        if NetworkConfig.GPU_ID >= 0:
            torch.cuda.set_device(NetworkConfig.GPU_ID)

        insts_before_split = train_insts  # self.prepare_instance_for_compilation(train_insts)

        insts = self.split_instances_for_train(insts_before_split)
        self._all_instances = insts

        self.touch(insts)

        # self._fm.get_param_g().lock_it()
        self.lock_it()

        # optimizer = torch.optim.SGD(self.parameters(), lr = 0.01)  # lr=0.8
        #print('self.parameters():', len(list(self.parameters()))
        optimizer = torch.optim.LBFGS(self.parameters())  # lr=0.8
        NetworkModel.Iter = 0
        for it in range(max_iterations):

            def closure():

                optimizer.zero_grad()

                all_loss = 0  ### scalar

                start_time = time.time()

                for i in range(len(self._all_instances)):
                    inst = self._all_instances[i]
                    if inst.get_instance_id() > 0:
                        network = self.get_network(i)
                        negative_network = self.get_network(i + 1)
                        network.nn_output = self._fm.build_nn_graph(inst)
                        negative_network.nn_output = network.nn_output

                for i in range(len(self._all_instances)):
                    loss = self.forward(self.get_network(i))
                    all_loss -= loss
                    # loss.backward()

                end_time = time.time()
                fwd_diff_time = end_time - start_time
                print('Forward:', '\tTime=', fwd_diff_time)

                start_time = time.time()
                all_loss.backward()
                end_time = time.time()
                bwd_diff_time = end_time - start_time
                print('Backward:', '\tTime=', bwd_diff_time)

                print(colored("Iteration ", 'yellow'), NetworkModel.Iter, ": Obj=", all_loss.item(), '\tTime=', fwd_diff_time + bwd_diff_time)
                NetworkModel.Iter += 1

                return all_loss

            # print('bWeight:', self.weights)
            # print("bGrad:", self.weights.grad)
            optimizer.step(closure)

            if NetworkModel.Iter > max_iterations:
                break
            # print('aWeight:', self.weights)
            # print("aGrad:", self.weights.grad)

    def forward(self, network):
        return network.inside()

    def get_network(self, network_id):


        if self._networks[network_id] != None:
            return self._networks[network_id]


        inst = self._all_instances[network_id]

        network = self._compiler.compile(network_id, inst, self._fm)
        # print("after compile: ", network)

        #if self._cache_networks:
        self._networks[network_id] = network

        return network

    def touch(self, insts):
        if self._networks == None:
            self._networks = [None for i in range(len(insts))]

        for network_id in range(len(insts)):
            if network_id % 100 == 0:
                eprint('.', end='')
            network = self.get_network(network_id)

            network.touch()

        eprint()

    def test(self, instances):
        return self.decode(instances=instances)

    def decode(self, instances, cache_features=False):

        self._num_threads = NetworkConfig.NUM_THREADS
        eprint('#Threads: ', self._num_threads)

        self._all_instances = instances

        # self._param = LocalNetworkParam(self, self._fm, len(instances))
        # self._fm.set_local_param(self._param)



        instances_output = []

        for k in range(len(instances)):
            instance = instances[k]
            # print("decode ", instance.is_labeled)

            network = self._compiler.compile(k, instance, self._fm)
            network.nn_output = self._fm.build_nn_graph(instance)
            network.max()
            instance_output = self._compiler.decompile(network)
            instances_output.append(instance_output)

        return instances_output




