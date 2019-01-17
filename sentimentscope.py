from hypergraph.NetworkCompiler import NetworkCompiler
from hypergraph.NetworkIDMapper import NetworkIDMapper
from hypergraph.TensorBaseNetwork import TensorBaseNetwork
from hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from hypergraph.NeuralBuilder import NeuralBuilder
from hypergraph.NetworkModel import NetworkModel
import torch.nn as nn
from hypergraph.Utils import *
from common.LinearInstance import LinearInstance
from hypergraph.Visualizer import Visualizer
import re
from termcolor import colored
from enum import Enum
import functools
from common.eval import *



START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
PAD = "<PAD>"
# SINK = "<SINK>"
# ROOT = "<ROOT>"

polar2symbol = {'positive':'+', 'neutral':'0', 'negative':'-'}
symbol2polar = {'+':'positive', '0':'neutral', '-':'negative'}

volitionnal_entity_type = ['PERSON', 'ORGANIZATION']
DEBUG = False

class NodeType(Enum):
    sink = 0
    scope = 1
    root = 2


class TSNetworkCompiler(NetworkCompiler):

    def __init__(self, label2id, labels, scope_tags, polar_tags, max_size=200):
        super().__init__()

        self.label2id = label2id
        self.labels = labels


        self.scope_tags = scope_tags
        self.polar_tags = polar_tags

        self.max_size = max_size
        ### position, tag, nodeType
        NetworkIDMapper.set_capacity(np.asarray([self.max_size, NodeType.root.value + 1, len(self.labels) ], dtype=np.int64))

        if DEBUG:
            print('TSNetworkCompiler:__init__()')
            print(self.label2id)
            print(self.labels)

    def to_sink(self):
        return self.to_node(0, 0, NodeType.sink.value)

    def to_scope(self, pos, label_id):
        return self.to_node(pos, label_id, NodeType.scope.value)

    def to_root(self, size):
        return self.to_node(size, 0, NodeType.root.value)

    def to_node(self, pos, label_id, node_type):
        return NetworkIDMapper.to_hybrid_node_ID(np.asarray([pos, node_type, label_id]))

    def start_of_entity(self, pos, size, output) -> bool:
        label = output[pos]

        if label.startswith('B'):
            return True

        if label.startswith('I') and (pos == 0 or output[pos - 1].startswith('O')):
            return True

        return False

    def end_of_entity(self, pos, size, output) -> bool:
        label = output[pos]

        if (not label.startswith('O')) and (pos == size - 1 or (not output[pos + 1].startswith('I'))):
            return True

        return False


    def get_targets(self, output) -> list:
        targets = []
        start_idx = 0
        size = len(output)

        for pos in range(len(output)):
            start_entity = self.start_of_entity(pos, size, output)
            end_entity = self.end_of_entity(pos, size, output)

            if start_entity:
                start_idx = pos

            if end_entity:
                label = output[pos][1]
                target = (start_idx, pos + 1, label)
                targets.append(target)

        return targets


    def compile_labeled(self, network_id, inst, param):

        builder = TensorBaseNetwork.NetworkBuilder.builder()

        node_sink = self.to_sink()
        builder.add_node(node_sink)

        node_root = self.to_root(inst.size())
        builder.add_node(node_root)

        size = inst.size()

        gold = inst.get_output()

        targets = self.get_targets(gold)

        for pos in range(size):
            for polor in self.polar_tags:
                for scope_tag in self.scope_tags:
                    node_scope = self.to_scope(pos, label2id[scope_tag + polor])
                    builder.add_node(node_scope)

        def add_edge(from_node, to_node_list):
            builder.add_node(from_node)
            builder.add_edge(from_node, to_node_list)


        for idx, target in enumerate(targets):

            s, e, polar = target
            target_len = e - s

            if idx == 0:

                from_node = node_sink

                for pos in range(0, s):
                    to_node = from_node
                    from_node = self.to_scope(pos, self.label2id['BB' + polar])
                    add_edge(from_node, [to_node])



                to_node = from_node
                from_node = self.to_scope(s, self.label2id['Be' + polar])
                add_edge(from_node, [to_node])


            else:

                last_entity_end_node = from_node  #eE or eS

                last_target = targets[idx - 1]
                last_s, last_e, last_polar = last_target


                scope_node = self.to_scope(s - 1, self.label2id['AB' + last_polar + polar])
                builder.add_node(scope_node)

                for pos in range(last_e, s):
                    scope_node = self.to_scope(pos - 1, self.label2id['AA' + last_polar])
                    builder.add_node(scope_node)

                    scope_node = self.to_scope(pos - 1, self.label2id['AB' + last_polar + polar])
                    builder.add_node(scope_node)

                    scope_node = self.to_scope(pos, self.label2id['BB' + polar])
                    builder.add_node(scope_node)

                #AA - AA
                from_node = last_entity_end_node
                for pos in range(last_e, s):
                    to_node = from_node
                    from_node = self.to_scope(pos - 1, self.label2id['AA' + last_polar])
                    builder.add_edge(from_node, [to_node])

                #AA - AB

                from_node = self.to_scope(last_e - 1, self.label2id['AB' + last_polar + polar])
                builder.add_edge(from_node, [last_entity_end_node])
                for pos in range(last_e, s):
                    to_node = self.to_scope(pos - 1, self.label2id['AA' + last_polar])
                    from_node = self.to_scope(pos, self.label2id['AB' + last_polar + polar])
                    builder.add_edge(from_node, [to_node])


                # AB - BB,  BB - BB
                for pos in range(last_e, s):
                    to_node = self.to_scope(pos - 1, self.label2id['AB' + last_polar + polar])
                    from_node = self.to_scope(pos, self.label2id['BB' + polar])
                    builder.add_edge(from_node, [to_node])

                    if pos > last_e:
                        to_node = self.to_scope(pos - 1, self.label2id['BB' + polar])
                        from_node = self.to_scope(pos, self.label2id['BB' + polar])
                        builder.add_edge(from_node, [to_node])



                from_node = self.to_scope(s, self.label2id['Be' + polar])
                builder.add_node(from_node)

                to_node = self.to_scope(s - 1, self.label2id['AB' + last_polar + polar])
                builder.add_edge(from_node, [to_node])

                if s - last_e > 0:
                    to_node = self.to_scope(s - 1, self.label2id['BB' + polar])
                    builder.add_edge(from_node, [to_node])




            if target_len == 1:
                to_node = from_node
                from_node = self.to_scope(s, self.label2id['eS' + polar])
                add_edge(from_node, [to_node])

            else:
                to_node = from_node
                from_node = self.to_scope(s, self.label2id['eB' + polar])
                add_edge(from_node, [to_node])

                for pos in range(s + 1, e - 1):
                    to_node = from_node
                    from_node = self.to_scope(pos, self.label2id['eM' + polar])
                    add_edge(from_node, [to_node])

                to_node = from_node
                from_node = self.to_scope(e - 1, self.label2id['eE' + polar])
                add_edge(from_node, [to_node])


        last_target = targets[-1]
        last_s, last_e, last_polar = last_target

        for pos in range(last_e - 1, size):
            to_node = from_node
            from_node = self.to_scope(pos, self.label2id['AA' + last_polar])
            add_edge(from_node, [to_node])

        to_node = from_node
        from_node = node_root
        add_edge(from_node, [to_node])


        network = builder.build(network_id, inst, param, self)
        return network

    def compile_unlabeled(self, network_id, inst, param):
        # return self.compile_labeled(network_id, inst, param)
        builder = TensorBaseNetwork.NetworkBuilder.builder()

        node_sink = self.to_sink()
        builder.add_node(node_sink)

        node_root = self.to_root(inst.size())
        builder.add_node(node_root)

        size = inst.size()
        label2id = self.label2id

        for pos in range(size):
            for polor in self.polar_tags:
                for scope_tag in self.scope_tags:
                    node_scope = self.to_scope(pos, label2id[scope_tag + polor])
                    builder.add_node(node_scope)


        # sink to colunm 0
        for polar in self.polar_tags:
            to_node = node_sink
            from_node = self.to_scope(0, label2id['BB' + polar])
            builder.add_edge(from_node, [to_node])

            from_node = self.to_scope(0, label2id['Be' + polar])
            builder.add_edge(from_node, [to_node])


        for pos in range(size):
            for polar in self.polar_tags:
                if pos < size - 1:

                    if pos < size - 2:
                        to_node = self.to_scope(pos, label2id['BB' + polar])
                        from_node = self.to_scope(pos + 1, label2id['BB' + polar])
                        builder.add_edge(from_node, [to_node])

                    to_node = self.to_scope(pos, label2id['BB' + polar])
                    from_node = self.to_scope(pos + 1, label2id['Be' + polar])
                    builder.add_edge(from_node, [to_node])


                    to_node = self.to_scope(pos, label2id['Be' + polar])
                    from_node = self.to_scope(pos, label2id['eB' + polar])
                    builder.add_edge(from_node, [to_node])


                to_node = self.to_scope(pos, label2id['Be' + polar])
                from_node = self.to_scope(pos, label2id['eS' + polar])
                builder.add_edge(from_node, [to_node])


                if pos > 0:
                    to_node = self.to_scope(pos, label2id['eE' + polar])
                    from_node = self.to_scope(pos, label2id['AA' + polar])
                    builder.add_edge(from_node, [to_node])

                to_node = self.to_scope(pos, label2id['eS' + polar])
                from_node = self.to_scope(pos, label2id['AA' + polar])
                builder.add_edge(from_node, [to_node])

                if pos < size - 1:

                    for next_polar in self.polar_tags:

                        if pos > 0:
                            to_node = self.to_scope(pos, label2id['eE' + polar])
                            from_node = self.to_scope(pos, label2id['AB' + polar + next_polar])
                            builder.add_edge(from_node, [to_node])

                        to_node = self.to_scope(pos, label2id['eS' + polar])
                        from_node = self.to_scope(pos, label2id['AB' + polar + next_polar])
                        builder.add_edge(from_node, [to_node])


                    if pos < size - 2:
                        to_node = self.to_scope(pos, label2id['eB' + polar])
                        from_node = self.to_scope(pos + 1, label2id['eM' + polar])
                        builder.add_edge(from_node, [to_node])

                        if pos > 0:
                            to_node = self.to_scope(pos, label2id['eM' + polar])
                            from_node = self.to_scope(pos + 1, label2id['eM' + polar])
                            builder.add_edge(from_node, [to_node])

                    to_node = self.to_scope(pos, label2id['eB' + polar])
                    from_node = self.to_scope(pos + 1, label2id['eE' + polar])
                    builder.add_edge(from_node, [to_node])

                    if pos > 0:
                        to_node = self.to_scope(pos, label2id['eM' + polar])
                        from_node = self.to_scope(pos + 1, label2id['eE' + polar])
                        builder.add_edge(from_node, [to_node])


                    to_node = self.to_scope(pos, label2id['AA' + polar])
                    from_node = self.to_scope(pos + 1, label2id['AA' + polar])
                    builder.add_edge(from_node, [to_node])




                    if pos < size - 2:
                        for next_polar in self.polar_tags:
                            to_node = self.to_scope(pos, label2id['AA' + polar])
                            from_node = self.to_scope(pos + 1, label2id['AB' + polar + next_polar])
                            builder.add_edge(from_node, [to_node])


                            to_node = self.to_scope(pos, label2id['AB' + polar + next_polar])
                            from_node = self.to_scope(pos + 1, label2id['BB' + next_polar])
                            builder.add_edge(from_node, [to_node])

                    for next_polar in self.polar_tags:
                        to_node = self.to_scope(pos, label2id['AB' + polar + next_polar])
                        from_node = self.to_scope(pos + 1, label2id['Be' + next_polar])
                        builder.add_edge(from_node, [to_node])


        for polar in self.polar_tags:
            to_node = self.to_scope(size - 1, label2id['AA' + polar])
            from_node = node_root
            builder.add_edge(from_node, [to_node])


        network = builder.build(network_id, inst, param, self)
        return network



    def decompile(self, network):
        inst = network.get_instance()

        size = inst.size()
        root_node = self.to_root(size)
        all_nodes = network.get_all_nodes()
        root_idx = np.argwhere(all_nodes == root_node)[0][0]  # network.count_nodes() - 1 #self._all_nodes.index(root_node)

        #curr_ = network.get_max_path(root_idx)[0]  # children[0]: root node
        curr_idx = root_idx

        pred_scope_tags_seq = []

        while True:
            curr_idx = network.get_max_path(curr_idx)[0]
            node_arr = network.get_node_array(curr_idx)
            pos, node_type, label_id = node_arr

            node_type_str = NodeType(node_type)
            label = self.labels[label_id]

            if node_type == NodeType.sink.value:
                break
            else:  # node_type == NodeType.Scope
                pred_scope_tags_seq.append(node_arr)


        pred_scope_tags_seq = pred_scope_tags_seq[::-1]

        prediction = []
        pred_sentiment_scope_split = []

        for idx, node_arr in enumerate(pred_scope_tags_seq):

            pos, node_type, label_id  = node_arr
            label = self.labels[label_id]

            if idx == len(pred_scope_tags_seq) - 1:
                next_node_arr = None
                next_label = 'X'
            else:
                next_node_arr = pred_scope_tags_seq[idx + 1]
                next_pos, next_node_type, next_label_id = next_node_arr
                next_label = self.labels[next_label_id]

            if label[0] == 'B':
                if next_label[0] == 'B':
                    prediction.append('O')
                elif next_label[0] == 'e':
                    polar = next_label[-1]
                    prediction.append('B' + polar)

            elif label[0] == 'e':
                if next_label[0] == 'e':
                    polar = label[-1]
                    prediction.append('I' + polar)
                else:
                    pass

            else: #A

                if pos < size - 1:

                    if next_label[0] == 'A':
                        prediction.append('O')
                    elif next_label[0] == 'B':
                        pred_sentiment_scope_split.append(pos)


        inst.set_prediction(prediction)
        inst.pred_sentiment_scope_split = pred_sentiment_scope_split
        return inst



class TSNeuralBuilder(NeuralBuilder):
    def __init__(self, gnp, labels, voc_size, word_embed_dim, lstm_dim = 100, dropout = 0.5):
        super().__init__(gnp)

        self.labels = labels
        self.label_size = len(labels)
        # self.word_embed = nn.Embedding(voc_size, self.token_embed, padding_idx=0).to(NetworkConfig.DEVICE)
        self.word_embed_dim = word_embed_dim
        self.lstm_dim = lstm_dim

        self.word_embeddings = nn.Embedding(voc_size, word_embed_dim).to(NetworkConfig.DEVICE)

        embed_dim = word_embed_dim
        self.rnn = nn.LSTM(embed_dim, lstm_dim, batch_first=True, bidirectional=True, dropout=dropout).to(NetworkConfig.DEVICE)

        self.sent = nn.Linear(lstm_dim * 2, 3)  # +, 0, -
        self.target = nn.Linear(lstm_dim * 2, 5) # O, B, M, E, S

        self.linear = nn.Linear(lstm_dim * 2, self.label_size)


    def load_pretrain(self, word2idx, path = None):
        emb = load_emb_glove(path, word2idx, self.word_embed_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(emb))
        self.word_embeddings = self.word_embeddings.to(NetworkConfig.DEVICE)


    def build_nn_graph(self, instance):

        size = instance.size()
        word_seq = instance.word_seq
        word_embs = self.word_embeddings(word_seq)

        word_rep = word_embs.unsqueeze(0) #torch.cat([word_embs], 1).unsqueeze(0)

        lstm_outputs, _ = self.rnn(word_rep, None)
        #lstm_outputs = lstm_outputs.squeeze(0)
        lstm_outputs = self.linear(lstm_outputs).squeeze(0)

        return lstm_outputs


    def get_nn_score(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        pos, node_type, label_id = parent_arr

        if node_type != NodeType.scope.value:  # Start, End
            return torch.tensor(0.0).to(NetworkConfig.DEVICE)
        else:

            return network.nn_output[pos][label_id]





    def get_label_id(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        pos, node_type, label_id = parent_arr
        if node_type == NodeType.scope.value:
            return label_id
        else:
            return -1


class TSReader():

    @staticmethod
    def read_insts(file, is_labeled, number):
        insts = []
        inputs = []
        outputs = []
        f = open(file, 'r', encoding='utf-8')
        for line in f:
            line = line.strip()

            if len(line) == 0:
                inst = LinearInstance(len(insts) + 1, 1, inputs, outputs)
                if is_labeled:
                    inst.set_labeled()
                else:
                    inst.set_unlabeled()
                insts.append(inst)

                inputs = []
                outputs = []

                if len(insts) >= number and number > 0:
                    break

            else:

                if line.startswith('## Tweet'):
                    continue

                fields = line.split()
                input = fields[0]
                input = re.sub('\d', '0', input)
                ner = fields[1]
                sent = fields[-1]

                if ner == 'O':
                    output = 'O'
                else:
                    ner_tag, ner_type = ner.split('-')

                    if ner_type not in volitionnal_entity_type:
                        output = 'O'

                    # try:
                    #     assert ner_type in volitionnal_entity_type
                    # except:
                    #     print(inputs)
                    #     print(outputs)
                    #     print(fields)
                    #     print('ner_type:',ner_type)
                    #     exit()
                    else:
                        if sent == '_':
                            sent = 'neutral'  #TODO: shall we remove it?

                        sent_symbol = polar2symbol[sent]

                        output = ner_tag + sent_symbol


                inputs.append(input)
                outputs.append(output)

        f.close()

        return insts


class TScore(Score):
    def __init__(self):
        self.target_fscore = FScore(0.0, 0.0, 0.0)
        self.sentiment_fscore = FScore(0.0, 0.0, 0.0)

    def set_scores(self, target_fscore: FScore, sentiment_fscore: FScore):
        self.target_fscore = target_fscore
        self.sentiment_fscore = sentiment_fscore

    def __str__(self):
        target_score =  str(self.target_fscore)
        sentiment_score = str(self.sentiment_fscore)
        return 'Target: ' + target_score + '\tSent:' + sentiment_score

    def to_tuple(self):
        return self.sentiment_fscore.to_tuple()

    def larger_than(self, obj):
        target_score_diff = self.target_fscore.fscore - obj.target_fscore.fscore
        return self.sentiment_fscore.fscore > obj.sentiment_fscore.fscore if math.fabs(target_score_diff) < 1e-8 else target_score_diff > 0

    def update_score(self, obj):
        self.target_fscore.update_score(obj.target_fscore)
        self.sentiment_fscore.update_score(obj.sentiment_fscore)



class sentimentscope_eval(Eval):

    def eval_by_script(self, output_filename: str) -> FScore:
        cmdline = ["perl", "scripts/conlleval.pl"]
        cmd = subprocess.Popen(cmdline, stdin=open(output_filename, 'r', encoding='utf-8'), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        stdout, stderr = cmd.communicate()

        ret =  stdout.decode("utf-8").split('\n')
        #print(stdout.decode("utf-8"))

        p, r, f1 = 0.0, 0.0, 0.0

        for line in ret:
            if line.startswith('accuracy'):
                # print('line:', line)
                for item in line.strip().split(';'):
                    items = item.strip().split(':')
                    if items[0].startswith('precision'):
                        p = float(items[1].strip()[:-1]) / 100
                    elif items[0].startswith('recall'):
                        r = float(items[1].strip()[:-1]) / 100
                    elif items[0].startswith('FB1'):
                        f1 = float(items[1].strip()) / 100

        fscore = FScore(p, r, f1)
        return fscore


    def eval(self, insts):

        result_target_file = 'tmp/result_target.txt'
        result_sent_file = 'tmp/result_sent.txt'

        f_target = open(result_target_file, 'w', encoding='utf-8')
        f_sent = open(result_sent_file, 'w', encoding='utf-8')

        for inst in insts:
            input = inst.get_input()

            gold = inst.get_output()
            pred = inst.get_prediction()

            target_gold = [item[0] for item in gold]
            target_pred = [item[0] for item in pred]

            sent_gold = [item[0] + '-' + symbol2polar[item[1]] if item[0][0] != 'O' else 'O' for item in gold]
            sent_pred = [item[0] + '-' + symbol2polar[item[1]] if item[0][0] != 'O' else 'O' for item in pred]

            target_lines = list(zip(input, target_gold, target_pred))
            sent_lines = list(zip(input, sent_gold, sent_pred))

            inst_target_str = '\n'.join(['\t'.join(item) for item in target_lines])
            inst_sent_str = '\n'.join(['\t'.join(item) for item in sent_lines])

            f_target.write(inst_target_str + '\n\n')
            f_sent.write(inst_sent_str + '\n\n')

        f_target.close()
        f_sent.close()


        target_ret = self.eval_by_script(result_target_file)
        sent_ret = self.eval_by_script(result_sent_file)

        ts_score = TScore()
        ts_score.set_scores(target_ret, sent_ret)

        return ts_score


class TSVisualizer(Visualizer):
    def __init__(self, compiler, fm, labels):
        super().__init__(compiler, fm)
        self.labels = labels
        self.span = 50


    def nodearr2label(self, node_arr):
        if node_arr[1] == 1:
            label_id = node_arr[2]
            label = self.labels[label_id]

            if label == 'eM0':
                return self.input[node_arr[0]]
            else:
                return self.labels[node_arr[2]] #+ ' ' + str(node_arr)
        else:
            if node_arr[1] == 0:
                return "<X>"
            else:
                return "<Root>"


    def nodearr2color(self, node_arr):
        if node_arr[1] == 1:
            label_id = node_arr[2]
            label = self.labels[label_id]
            if label[0] == 'B':
                return 'purple'
            elif label[0] == 'e':
                return 'yellow'
            else:
                return 'green'
        else:
            return 'blue'


    def nodearr2coord(self, node_arr):
        span = self.span

        if node_arr[1] == 1:
            x = node_arr[0] * span
            label_id = node_arr[2]
            label = self.labels[label_id]
            scope_tag = label[:2]
            polar = label[2]

            if polar == '+':
                polar_id = 2
            elif polar == '0':
                polar_id = 1
            else:
                polar_id = 0

            y = (polar_id + 1) * span

            if label[0] == 'B':
                x -= 15
                y += 4
                if label[1] == 'B':
                    y += 6

            elif label[0] == 'A':
                x += 15
                y -= 4
                if label[1] == 'B':
                    y -= 6

                    if label[3] == '+':
                        y -= 0
                    elif label[3] == '0':
                        y -= 6
                    else: #label[2] == '-':
                        y -= 12

            else: #label[0] == 'e':
                if label[1] == 'B':
                    y += 6
                elif label[1] == 'M':
                    y += 2
                elif label[1] == 'E':
                    y -= 2
                else:
                    y -= 6


            return (x, y)


        else:
            if node_arr[1] == 0:
                return (-1 * span , 0.0)
            else:
                return (node_arr[0] * span, 0.0)






if __name__ == "__main__":

    torch.manual_seed(9997)
    np.random.seed(9997)
    torch.set_num_threads(40)


    train_file = "data/ts/train.1.conll.train_test"
    dev_file = "data/ts/test.1.conll.train_test"
    test_file = "data/ts/test.1.conll.train_test"
    trial_file = "data/ts/trial.txt"

    import os
    from random import shuffle
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    TRIAL = False
    num_train = -1
    num_dev = -1
    num_test = -1
    num_iter = 40
    batch_size = 1
    num_thread = 1
    dev_file = test_file
    NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH = True
    NetworkConfig.GPU_ID = -1
    embed_path = 'embedding/glove.6B.100d.txt'
    word_embed_dim = 100
    APPEND_START_END = False
    visual = True
    DEBUG = False

    if TRIAL == True:
        data_size = -1
        train_file = trial_file
        dev_file = trial_file
        test_file = trial_file

    if NetworkConfig.GPU_ID > -1:
        NetworkConfig.DEVICE = torch.device("cuda:" + str(NetworkConfig.GPU_ID))

    if num_thread > 1:
        NetworkConfig.NUM_THREADS = num_thread
        print('Set NUM_THREADS = ', num_thread)

    train_insts = TSReader.read_insts(train_file, True, num_train)
    dev_insts = TSReader.read_insts(dev_file, False, num_dev)
    test_insts = TSReader.read_insts(test_file, False, num_test)


    vocab2id = {PAD:0, UNK:1}
    label2id = {}
    scope_tags = ['BB', 'Be', 'eB', 'eM', 'eE', 'eS', 'AA', 'AB+', 'AB0', 'AB-']
    polar_tags = ['+', '0', '-']

    for scope_tag in scope_tags:
        for polar in polar_tags:
            label2id[scope_tag + polar] = len(label2id)


    labels = [()] * (len(label2id))  ##  include (), but not dummy label
    for key in label2id:
        labels[label2id[key]] = key

    if DEBUG:
        print('label2id:',list(label2id.keys()))


    for inst in train_insts:  #+ dev_insts + test_insts:
        for word in inst.input:
            if word not in vocab2id:
                vocab2id[word] = len(vocab2id)


    for inst in train_insts: # + dev_insts + test_insts:
        if APPEND_START_END:
            input_seq = [(START, START)] + inst.input + [(STOP, STOP)]
        else:
            input_seq = inst.input

        inst.word_seq = torch.tensor([vocab2id[word] for word in input_seq]).to(NetworkConfig.DEVICE)

    # old_train_insts = train_insts
    # shuffle(old_train_insts)


    for inst in dev_insts + test_insts:
        if APPEND_START_END:
            input_seq = [(START, START)] + inst.input + [(STOP, STOP)]
        else:
            input_seq = inst.input

        inst.word_seq = torch.tensor([vocab2id[word] if word in vocab2id else vocab2id[UNK] for word in input_seq]).to(NetworkConfig.DEVICE)


    gnp = TensorGlobalNetworkParam()
    neural_builder = TSNeuralBuilder(gnp, labels, len(vocab2id), word_embed_dim)
    neural_builder.load_pretrain(vocab2id, embed_path)
    compiler = TSNetworkCompiler(label2id, labels, scope_tags, polar_tags)
    evaluator = sentimentscope_eval()
    model = NetworkModel(neural_builder, compiler, evaluator)
    model.model_path = "best_sentimentscope.pt"
    #model.set_visualizer(ts_visualizer)


    if DEBUG:
        if visual:
            ts_visualizer = TSVisualizer(compiler, neural_builder, labels)
            inst = train_insts[0]
            inst.is_labeled = False
            ts_visualizer.visualize_inst(inst)
            #inst.is_labeled = False
            #ts_visualizer.visualize_inst(inst)
            exit()

    if batch_size == 1:
        model.learn(train_insts, num_iter, dev_insts, test_insts)
    else:
        model.learn_batch(train_insts, num_iter, dev_insts, batch_size)

    try:
        model.load()

        results = model.test(test_insts)
        for inst in results:
            print(inst.get_input())
            print(inst.get_output())
            print(inst.get_prediction())
            print()

        ret = model.evaluator.eval(results)
        print(ret)
    except:
        print('Loading model fails ...')


