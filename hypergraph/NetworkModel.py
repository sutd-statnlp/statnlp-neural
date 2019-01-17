import torch.nn as nn
import torch.optim
from hypergraph.Utils import *
import time
from termcolor import colored
import copy
from multiprocessing import Process

class NetworkModel(nn.Module):
    Iter = 0

    def __init__(self, fm, compiler, evaluator):
        super().__init__()
        self.fm = fm
        self.compiler = compiler
        self.all_instances = None
        self.all_instances_test = None
        self.networks = None
        self.networks_test = None
        self.evaluator = evaluator
        self.model_path = 'best_model.pt'

    def set_model_path(self, path):
        self.model_path = path

    def get_instances(self):
        return self.all_instances

    def get_feature_manager(self):
        return self.fm

    def get_network_compiler(self):
        return self.compiler

    def split_instances_for_train(self, insts_before_split):
        print("#instances=", len(insts_before_split))
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
        gnp = self.fm.get_param_g()

        if gnp.is_locked():
            return

        gnp.finalize_transition()
        gnp.locked = True


    def learn_batch(self, train_insts, max_iterations, dev_insts, batch_size = 10):

        insts_before_split = train_insts

        insts = self.split_instances_for_train(insts_before_split)
        self.all_instances = insts



        self.touch(self.all_instances)

        label_networks = []
        unlabel_networks = []
        for i in range(0, len(self.all_instances), 2):
            label_networks.append(self.get_network(i))
            unlabel_networks.append(self.get_network(i + 1))


        batches = self.fm.generate_batches(insts_before_split, batch_size)


        self.lock_it()


        optimizer = torch.optim.Adam(self.parameters())

        self.best_ret = [0, 0, 0]



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
                nn_output_batch = self.fm.build_nn_graph_batch(batch_input_seqs)

                batch_label_networks = label_networks[batch_network_id_range[0]:batch_network_id_range[1]]
                batch_unlabel_networks = unlabel_networks[batch_network_id_range[0]:batch_network_id_range[1]]

                this_batch_size = nn_output_batch.shape[0]

                for b in range(this_batch_size):
                    batch_label_networks[b].nn_output = nn_output_batch[b]
                    batch_unlabel_networks[b].nn_output = nn_output_batch[b]

                if NetworkConfig.NUM_THREADS == 1:
                    for b in range(this_batch_size):
                        label_score = self.forward(batch_label_networks[b])
                        unlabel_score = self.forward(batch_unlabel_networks[b])
                        loss = -unlabel_score - label_score
                        batch_loss += loss
                else:
                    num_thread = NetworkConfig.NUM_THREADS
                    tasks = list(range(0, this_batch_size))
                    num_per_basket = this_batch_size // num_thread if this_batch_size % num_thread == 0 else batch_size // num_thread + 1
                    task_ids_each_thread = [tasks[i:i + num_per_basket] for i in range(0, this_batch_size, num_per_basket)]
                    task_ids_each_thread += [[]] * (num_thread - len(task_ids_each_thread))

                    batch_loss_per_thread = [0] * num_thread

                    def calc_score_thread(task_ids, thread_idx):
                        for id in task_ids:
                            label_score = self.forward(batch_label_networks[id])
                            unlabel_score = self.forward(batch_unlabel_networks[id])
                            loss = -unlabel_score - label_score
                            batch_loss_per_thread[thread_idx] = batch_loss_per_thread[thread_idx] + loss


                    processes = []
                    for thread_idx in range(NetworkConfig.NUM_THREADS):
                        p = Process(target=calc_score_thread(task_ids_each_thread[thread_idx], thread_idx))
                        processes.append(p)
                        p.start()

                    for thread_idx in range(NetworkConfig.NUM_THREADS):
                        processes[thread_idx].join()

                    for thread_idx in range(NetworkConfig.NUM_THREADS):
                        batch_loss += batch_loss_per_thread[thread_idx]


                batch_loss.backward()
                optimizer.step()

                all_loss += batch_loss.item()
                #print(colored("Batch {0}".format(batch_idx), 'yellow'), iteration, ": batch loss =", batch_loss.item(), flush=True)


            end_time = time.time()

            print(colored("Epoch ", 'red'), iteration, ": Obj=", all_loss, '\tTime=', end_time - start_time, flush=True)


            start_time = time.time()
            self.decode(dev_insts)
            ret = self.evaluator.eval(dev_insts)
            end_time = time.time()
            print(ret, '\tTime=', end_time - start_time, flush=True)
            print()

            if self.best_ret[2] < ret[2]:
                self.best_ret = ret
                self.save()

        print("Best F1:", self.best_ret)


    def learn(self, train_insts, max_iterations, dev_insts, test_insts, optimizer = 'adam'):

        if optimizer == "lbfgs":
            self.learn_lbfgs(train_insts, max_iterations, dev_insts)
            return

        insts_before_split = train_insts

        insts = self.split_instances_for_train(insts_before_split)
        self.all_instances = insts

        self.touch(insts)

        self.lock_it()

        parameters = filter(lambda p: p.requires_grad, self.parameters())

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters,lr=NetworkConfig.NEURAL_LEARNING_RATE)
        else:
            print(colored('Unsupported optimizer:', 'red'), optimizer)
            return

        self.best_score = None


        print('Start Training...', flush=True)
        for iteration in range(max_iterations):
            self.train()
            all_loss = 0
            start_time = time.time()
            for i in range(len(self.all_instances)):
                inst = self.all_instances[i]
                if inst.get_instance_id() > 0:
                    optimizer.zero_grad()
                    self.zero_grad()
                    network = self.get_network(i)
                    negative_network = self.get_network(i + 1)

                    if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
                        network.touch()
                        negative_network.touch()

                    network.nn_output = self.fm.build_nn_graph(inst)
                    negative_network.nn_output = network.nn_output

                    label_score = self.forward(network)
                    unlabel_score = self.forward(negative_network)
                    loss = -unlabel_score - label_score
                    all_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                    if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
                        del network
                        self.networks[i] = None
                        del negative_network
                        self.networks[i + 1] = None
                # if (i + 1) % 100 == 0:
                #     print('.', end='')
            print()
            end_time = time.time()

            print(colored("Iteration ", 'yellow'), iteration, ": Obj=", all_loss, '\tTime={:.2f}s'.format(end_time - start_time), flush=True)

            start_time = time.time()
            self.decode(dev_insts)
            score = self.evaluator.eval(dev_insts)
            end_time = time.time()
            print(str(score), '\tTime={:.2f}s'.format(end_time - start_time), flush=True)


            if self.best_score == None or score.larger_than(self.best_score):

                if self.best_score == None:
                    self.best_score = score
                else:
                    self.best_score.update_score(score)
                self.save()

                start_time = time.time()
                self.decode(test_insts)
                test_score = self.evaluator.eval(test_insts)
                end_time = time.time()
                print("On Test -- ", str(test_score), '\tTime={:.2f}s'.format(end_time - start_time), flush=True)


        print("Best Result:", self.best_score)


    def learn_lbfgs(self, train_insts, max_iterations, dev_insts):


        insts_before_split = train_insts

        insts = self.split_instances_for_train(insts_before_split)
        self.all_instances = insts

        self.touch(insts)
        self.lock_it()

        optimizer = torch.optim.LBFGS(self.parameters())
        self.iteration = 0
        self.best_ret = [0, 0, 0]


        self.iteration = 0
        def closure():
            self.train()
            self.zero_grad()
            optimizer.zero_grad()

            all_loss = 0

            start_time = time.time()
            for i in range(len(self.all_instances)):
                inst = self.all_instances[i]
                if inst.get_instance_id() > 0:
                    network = self.get_network(i)
                    negative_network = self.get_network(i + 1)
                    network.nn_output = self.fm.build_nn_graph(inst)
                    negative_network.nn_output = network.nn_output


            for i in range(len(self.all_instances)):
                loss = self.forward(self.get_network(i))
                all_loss -= loss



            all_loss.backward()
            end_time = time.time()

            print(colored("Iteration ", 'yellow'), self.iteration, ": Obj=", all_loss.item(), '\tTime=', end_time - start_time, flush=True)

            start_time = time.time()
            self.decode(dev_insts)
            ret = self.evaluator.eval(dev_insts)
            end_time = time.time()
            print("Prec.: {0:.2f} Rec.: {1:.2f} F1.: {2:.2f}".format(ret[0], ret[1], ret[2]), '\tTime={:.2f}s'.format(end_time - start_time), flush=True)


            if self.best_ret[2] < ret[2]:
                self.best_ret = ret
                self.save()

            self.iteration += 1
            if self.iteration >= max_iterations:
                return 0

            return all_loss


        while self.iteration < max_iterations:
            optimizer.step(closure)


        print("Best Result:", self.best_ret)

    def forward(self, network):
        return network.inside()

    def get_network(self, network_id):

        if self.networks[network_id] != None:
            return self.networks[network_id]

        inst = self.all_instances[network_id]

        network = self.compiler.compile(network_id, inst, self.fm)
        self.networks[network_id] = network

        return network


    def touch(self, insts):
        print('Touching ...', flush=True)
        if self.networks == None:
            self.networks = [None for i in range(len(insts))]

        if NetworkConfig.IGNORE_TRANSITION:
            print('Ignore Transition...')
            return


        start_time = time.time()

        num_thread = NetworkConfig.NUM_THREADS

        if num_thread > 1:
            print('Multi-thread Touching...')
            num_networks = len(insts)
            num_per_bucket = num_networks // num_thread if num_networks % num_thread == 0 else num_networks // num_thread + 1


            def touch_networks(bucket_id):
                end = num_per_bucket * (bucket_id + 1)
                end = min(num_networks, end)
                counter = 1
                for network_id in range(num_per_bucket * bucket_id, end):
                    if counter % 100 == 0:
                        print('.', end='', flush=True)
                    network = self.get_network(network_id)
                    network.touch()
                    counter += 1

                    if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
                        del network


            processes = []
            for thread_idx in range(num_thread):
                p = Process(target=touch_networks(thread_idx))
                processes.append(p)
                p.start()

            for thread_idx in range(num_thread):
                processes[thread_idx].join()

        else:
            for network_id in range(len(insts)):
                if network_id % 100 == 0:
                    print('.', end='', flush=True)
                network = self.get_network(network_id)

                network.touch()

                if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
                    del network

        end_time = time.time()

        print(flush=True)
        print('Toucing Completes taking ', end_time - start_time, ' seconds.', flush=True)

        if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
            del self.networks
            self.networks = [None] * len(insts)


    def test(self, instances):
        return self.decode(instances=instances)

    def decode(self, instances, cache_features=False):

        self.all_instances_test = instances
        self.eval()
        instances_output = []

        for k in range(len(instances)):
            instance = instances[k]

            network = self.compiler.compile(k, instance, self.fm)
            network.touch()
            network.nn_output = self.fm.build_nn_graph(instance)
            network.max()
            instance_output = self.compiler.decompile(network)
            instances_output.append(instance_output)

        return instances_output


    def get_network_test(self, network_id):
        if self.networks_test[network_id] != None:
            return self.networks_test[network_id]

        inst = self.all_instances_test[network_id]

        network = self.compiler.compile(network_id, inst, self.fm)


        self.networks_test[network_id] = network

        return network


    def touch_test(self, insts):
        if self.networks_test == None:
            self.networks_test = [None for i in range(len(insts))]

        for network_id in range(len(insts)):
            if network_id % 100 == 0:
                print('.', end='')
            network = self.get_network_test(network_id)

            network.touch()

        print()


    def save(self):
        torch.save(self.state_dict(), self.model_path)
        print(colored('Save the best model to ', 'red'), self.model_path)

    def load(self):
        self.load_state_dict(torch.load(self.model_path))


    def set_visualizer(self, visualizer):
        self.visualizer = visualizer


    def visualize(self, network_id):
        network = self.get_network(network_id)
        self.visualizer.visualize(network)