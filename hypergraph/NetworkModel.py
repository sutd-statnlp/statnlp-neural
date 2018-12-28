import torch.nn as nn
import torch.optim
from hypergraph.Utils import *
import time
from termcolor import colored
import copy

class NetworkModel(nn.Module):
    Iter = 0

    def __init__(self, fm, compiler, evaluator):
        super().__init__()
        self._fm = fm
        self._compiler = compiler
        self._all_instances = None
        self._all_instances_test = None
        self._networks = None
        self._networks_test = None
        self.evaluator = evaluator

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


    def split_instances_for_train_two(self, insts_before_split):
        eprint("#instances=", len(insts_before_split))
        label_insts = []
        unlabel_insts = []

        k = 0
        for i in range(len(insts_before_split)):
            label_inst = insts_before_split[k]
            unlabel_inst = label_inst.duplicate()
            unlabel_inst.set_instance_id(-label_inst.get_instance_id())
            unlabel_inst.set_weight(-label_inst.get_weight())
            unlabel_inst.set_unlabeled()

            label_insts.append(label_inst)
            unlabel_insts.append(unlabel_inst)
            k = k + 1

        return label_insts, unlabel_insts


    def lock_it(self):
        gnp = self._fm.get_param_g()

        if gnp.is_locked():
            return

        gnp.finalize_transition()
        gnp.locked = True


    def learn_batch(self, train_insts, max_iterations, dev_insts, batch_size = 10):

        insts_before_split = train_insts  # self.prepare_instance_for_compilation(train_insts)

        insts = self.split_instances_for_train(insts_before_split)
        self._all_instances = insts

        #self._all_instances = label_insts + unlabel_insts

        self.touch(self._all_instances)

        label_networks = []
        unlabel_networks = []
        for i in range(0, len(self._all_instances), 2):
            label_networks.append(self.get_network(i))
            unlabel_networks.append(self.get_network(i + 1))


        batches = self._fm.generate_batches(insts_before_split, batch_size)

        # self._fm.get_param_g().lock_it()
        self.lock_it()

        # optimizer = torch.optim.SGD(self.parameters(), lr = 0.01)  # lr=0.8
        # print('self.parameters():', len(list(self.parameters()))
        # optimizer = torch.optim.LBFGS(self.parameters())  # lr=0.8
        optimizer = torch.optim.Adam(self.parameters())
        #NetworkModel.Iter = 0
        self.best_ret = [0, 0, 0]
        best_model = None
        # for it in range(max_iterations):

        # self.iteration = 0

        print('Start Training...', flush=True)
        for iteration in range(max_iterations):
            self.train()
            all_loss = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(batches):
                optimizer.zero_grad()
                self.zero_grad()

                batch_loss = 0

                batch_input_seqs, batch_network_id_range = batch
                nn_output_batch = self._fm.build_nn_graph_batch(batch_input_seqs)

                batch_label_networks = label_networks[batch_network_id_range[0]:batch_network_id_range[1]]
                batch_unlabel_networks = unlabel_networks[batch_network_id_range[0]:batch_network_id_range[1]]


                for b in range(nn_output_batch.shape[0]):
                    batch_label_networks[b].nn_output = nn_output_batch[b]
                    batch_unlabel_networks[b].nn_output = nn_output_batch[b]

                    label_score = self.forward(batch_label_networks[b])
                    unlabel_score = self.forward(batch_unlabel_networks[b])
                    loss = -unlabel_score - label_score
                    batch_loss += loss

                batch_loss.backward()
                optimizer.step()

                all_loss += batch_loss.item()
                #print(colored("Batch {0}".format(batch_idx), 'yellow'), iteration, ": batch loss =", batch_loss.item(), flush=True)


            end_time = time.time()

            print(colored("Epoch ", 'red'), iteration, ": Obj=", all_loss, '\tTime=', end_time - start_time, flush=True)
            #NetworkModel.Iter += 1

            start_time = time.time()
            self.decode(dev_insts)
            ret = self.evaluator.eval(dev_insts)
            end_time = time.time()
            print(ret, '\tTime=', end_time - start_time, flush=True)
            print()

            if self.best_ret[2] < ret[2]:
                self.best_ret = ret
                # best_model = copy.deepcopy(ner_model)

            # if iteration >= max_iterations:
            #     return 0

        print("Best F1:", self.best_ret)

    def learn(self, train_insts, max_iterations, dev_insts):

        if NetworkConfig.GPU_ID >= 0:
            torch.cuda.set_device(NetworkConfig.GPU_ID)

        insts_before_split = train_insts  # self.prepare_instance_for_compilation(train_insts)

        insts = self.split_instances_for_train(insts_before_split)
        self._all_instances = insts

        self.touch(insts)

        # self._fm.get_param_g().lock_it()
        self.lock_it()

        # optimizer = torch.optim.SGD(self.parameters(), lr = 0.01)  # lr=0.8
        # print('self.parameters():', len(list(self.parameters()))
        # optimizer = torch.optim.LBFGS(self.parameters())  # lr=0.8
        optimizer = torch.optim.Adam(self.parameters())
        #NetworkModel.Iter = 0
        self.best_ret = [0, 0, 0]
        best_model = None
        # for it in range(max_iterations):

        # self.iteration = 0

        print('Start Training...', flush=True)
        for iteration in range(max_iterations):
            self.train()
            all_loss = 0
            start_time = time.time()
            for i in range(len(self._all_instances)):
                inst = self._all_instances[i]
                if inst.get_instance_id() > 0:
                    optimizer.zero_grad()
                    self.zero_grad()
                    network = self.get_network(i)
                    negative_network = self.get_network(i + 1)
                    network.nn_output = self._fm.build_nn_graph(inst)
                    negative_network.nn_output = network.nn_output

                    label_score = self.forward(network)
                    unlabel_score = self.forward(negative_network)
                    loss = -unlabel_score - label_score
                    all_loss += loss.item()
                    loss.backward()
                    optimizer.step()

            end_time = time.time()

            print(colored("Iteration ", 'yellow'), iteration, ": Obj=", all_loss, '\tTime=', end_time - start_time, flush=True)
            #NetworkModel.Iter += 1

            start_time = time.time()
            self.decode(dev_insts)
            ret = self.evaluator.eval(dev_insts)
            end_time = time.time()
            print(ret, '\tTime=', end_time - start_time, flush=True)

            if self.best_ret[2] < ret[2]:
                self.best_ret = ret
                # best_model = copy.deepcopy(ner_model)

            # if iteration >= max_iterations:
            #     return 0

        print("Best F1:", self.best_ret)

    def learn_lbfgs(self, train_insts, max_iterations, dev_insts):

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
        self.best_ret = [0, 0, 0]
        best_model = None
        # for it in range(max_iterations):

        self.iteration = 0
        def closure():
            self.train()
            self.zero_grad()
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
            #end_time = time.time()
            #fwd_diff_time = end_time - start_time
            #print('Nueral Forward:', '\tTime=', fwd_diff_time)

            #start_time = time.time()
            for i in range(len(self._all_instances)):
                loss = self.forward(self.get_network(i))
                all_loss -= loss
                # loss.backward()

            #end_time = time.time()
            #fwd_diff_time = end_time - start_time
            #print('Forward:', '\tTime=', fwd_diff_time)

            #start_time = time.time()
            all_loss.backward()
            end_time = time.time()
            #bwd_diff_time = end_time - start_time
            #print('Backward:', '\tTime=', bwd_diff_time)

            print(colored("Iteration ", 'yellow'), NetworkModel.Iter, ": Obj=", all_loss.item(), '\tTime=', end_time - start_time, flush=True)
            NetworkModel.Iter += 1

            start_time = time.time()
            self.decode(dev_insts)
            ret = self.evaluator.eval(dev_insts)
            end_time = time.time()
            print(ret, '\tTime=',  end_time - start_time, flush=True)
            if self.best_ret[2] < ret[2]:
                self.best_ret = ret
                # best_model = copy.deepcopy(ner_model)

            self.iteration += 1
            if self.iteration >= max_iterations:
                return 0

            return all_loss



        optimizer.step(closure)



        print("Best F1:", self.best_ret)

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
        print('Touching...', flush=True)
        if self._networks == None:
            self._networks = [None for i in range(len(insts))]

        for network_id in range(len(insts)):
            if network_id % 100 == 0:
                print('.', end='', flush=True)
            network = self.get_network(network_id)

            network.touch()

        print(flush=True)




    def test(self, instances):
        return self.decode(instances=instances)

    def decode(self, instances, cache_features=False):

        # self._num_threads = NetworkConfig.NUM_THREADS
        # print('#Threads: ', self._num_threads)
        self._all_instances_test = instances
        self.eval()
        instances_output = []

        for k in range(len(instances)):
            instance = instances[k]
            # print("decode ", instance.is_labeled)

            network = self._compiler.compile(k, instance, self._fm)
            network.touch()
            network.nn_output = self._fm.build_nn_graph(instance)
            network.max()
            instance_output = self._compiler.decompile(network)
            instances_output.append(instance_output)

        return instances_output


    def get_network_test(self, network_id):
        if self._networks_test[network_id] != None:
            return self._networks_test[network_id]

        inst = self._all_instances_test[network_id]

        network = self._compiler.compile(network_id, inst, self._fm)
        # print("after compile: ", network)

        # if self._cache_networks:
        self._networks_test[network_id] = network

        return network


    def touch_test(self, insts):
        if self._networks_test == None:
            self._networks_test = [None for i in range(len(insts))]

        for network_id in range(len(insts)):
            if network_id % 100 == 0:
                print('.', end='')
            network = self.get_network_test(network_id)

            network.touch()

        print()