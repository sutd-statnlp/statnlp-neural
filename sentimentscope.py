from hypergraph.NetworkCompiler import NetworkCompiler
from hypergraph.NetworkIDMapper import NetworkIDMapper
from hypergraph.TensorBaseNetwork import TensorBaseNetwork
from hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from hypergraph.NeuralBuilder import NeuralBuilder
from hypergraph.NetworkModel import NetworkModel
import torch.nn as nn
from hypergraph.Utils import *
from common.LinearInstance import LinearInstance
import re
from termcolor import colored
from enum import Enum
import functools
from common.eval import *
import torch.nn.functional as F


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
    def __init__(self, gnp, labels, voc_size, word_embed_dim, postag_size, postag_embed_dim, char_emb_size, charlstm_hidden_dim, SENT_emb_size, SENT_embed_dim, THER_SENT_embed_dim, browncluster5_embed_dim, browncluster3_embed_dim, lstm_dim = 200, dropout = 0.5):
        super().__init__(gnp)

        self.labels = labels
        self.label_size = len(labels)
        # self.word_embed = nn.Embedding(voc_size, self.token_embed, padding_idx=0).to(NetworkConfig.DEVICE)
        self.word_embed_dim = word_embed_dim
        self.lstm_dim = lstm_dim
        self.postag_embed_dim = postag_embed_dim
        self.char_emb_size = char_emb_size
        self.SENT_embed_dim = SENT_embed_dim
        self.THER_SENT_embed_dim = THER_SENT_embed_dim
        self.browncluster5_embed_dim = browncluster5_embed_dim
        self.browncluster3_embed_dim = browncluster3_embed_dim


        self.linear_dim = lstm_dim * 2 + self.postag_embed_dim + self.SENT_embed_dim + self.THER_SENT_embed_dim +  self.browncluster5_embed_dim + self.browncluster3_embed_dim

        self.zero = torch.tensor(0.0).to(NetworkConfig.DEVICE)

        self.word_embeddings = nn.Embedding(voc_size, word_embed_dim).to(NetworkConfig.DEVICE)

        # if postag_embed_dim > 0:
        #     self.posttag_embeddings = nn.Embedding(postag_size, postag_embed_dim).to(NetworkConfig.DEVICE)
        #     tag_embed_parameter = np.empty([postag_size, postag_embed_dim])
        #     scale = np.sqrt(3.0 / postag_embed_dim)
        #     for i in range(postag_size):
        #         tag_embed_parameter[i] = np.random.uniform(-scale, scale, [1, postag_embed_dim])
        #     self.posttag_embeddings.weight.data.copy_(torch.from_numpy(tag_embed_parameter))

        if char_emb_size > 0:
            from features.char_lstm import CharBiLSTM
            self.char_bilstm = CharBiLSTM(char2id, chars, char_emb_size, charlstm_hidden_dim).to(NetworkConfig.DEVICE)


        # if SENT_embed_dim > 0:
        #     self.SENT_embeddings = nn.Embedding(SENT_emb_size, SENT_embed_dim).to(NetworkConfig.DEVICE)

        self.dropout = nn.Dropout(dropout).to(NetworkConfig.DEVICE)

        embed_dim = word_embed_dim + char_emb_size # + postag_embed_dim+ SENT_embed_dim

        self.rnn = nn.LSTM(embed_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=1).to(NetworkConfig.DEVICE)

        self.build_linear_layers()


    def build_linear_layers(self):
        self.linear = nn.Linear(self.linear_dim, self.label_size).to(NetworkConfig.DEVICE)


    def load_pretrain(self, word2idx, path = None):
        emb = load_emb_glove(path, word2idx, self.word_embed_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(emb))
        self.word_embeddings = self.word_embeddings.to(NetworkConfig.DEVICE)

    def build_features(self, instance):
        size = instance.size()
        word_seq = instance.word_seq.unsqueeze(0)
        word_embs = self.word_embeddings(word_seq)
        word_embs = self.dropout(word_embs)
        word_rep = [word_embs]

        # if self.postag_embed_dim > 0:
        #     postag_seq = instance.postag_seq.unsqueeze(0)
        #     postag_embs = self.posttag_embeddings(postag_seq)
        #     word_rep.append(postag_embs)

        if self.char_emb_size > 0:
            char_seq_tensor = instance.char_seq_tensor.unsqueeze(0)
            char_seq_len = instance.char_seq_len.unsqueeze(0)
            char_features = self.char_bilstm.get_last_hiddens(char_seq_tensor,
                                                              char_seq_len)  # batch_size, sent_len, char_hidden_dim
            word_rep.append(char_features)



        word_rep = torch.cat(word_rep, 2)

        lstm_outputs, _ = self.rnn(word_rep, None)

        lstm_outputs = lstm_outputs.squeeze(0)

        feature_output = [lstm_outputs]


        def create_discrete_feature(seq, embed_dim):
            if embed_dim > 0:
                seq = seq.unsqueeze(0).view(size, 1)
                discrete_embs = torch.zeros(size, embed_dim).to(NetworkConfig.DEVICE)
                discrete_embs.scatter_(1, seq, 1.0)
                feature_output.append(discrete_embs)
                #return discrete_embs


        create_discrete_feature(instance.postag_seq, self.postag_embed_dim)
        create_discrete_feature(instance.browncluster5_seq, self.browncluster5_embed_dim)
        create_discrete_feature(instance.browncluster3_seq, self.browncluster3_embed_dim)
        create_discrete_feature(instance.SENT_seq, self.SENT_embed_dim)
        create_discrete_feature(instance.THER_SENT_seq, self.THER_SENT_embed_dim)


        feature_output = torch.cat(feature_output, 1)

        return feature_output

    def build_nn_graph(self, instance):
        lstm_outputs = self.build_features(instance)
        nn_output = self.linear(lstm_outputs)
        return nn_output


    def get_nn_score(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        pos, node_type, label_id = parent_arr

        if node_type != NodeType.scope.value:  # Start, End
            return self.zero#torch.tensor(0.0).to(NetworkConfig.DEVICE)
        else:
            return network.nn_output[pos][label_id]



    def get_label_id(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        pos, node_type, label_id = parent_arr
        if node_type == NodeType.scope.value:
            return label_id
        else:
            return -1



class TSATTNeuralBuilder(TSNeuralBuilder):
    def __init__(self, gnp, labels, voc_size, word_embed_dim, postag_size, postag_embed_dim, char_emb_size, charlstm_hidden_dim, SENT_emb_size, SENT_embed_dim, THER_SENT_embed_dim, browncluster5_embed_dim, browncluster3_embed_dim, lstm_dim = 200, dropout = 0.5):
        super().__init__(gnp, labels, voc_size, word_embed_dim, postag_size, postag_embed_dim, char_emb_size, charlstm_hidden_dim, SENT_emb_size, SENT_embed_dim, THER_SENT_embed_dim, browncluster5_embed_dim, browncluster3_embed_dim, lstm_dim, dropout)
        print('[Info] TSATTNeuralBuilder is initialized')

    def init_attention(self, attention_type, attention_size = 100):
        self.attention_type = attention_type
        self.attention_size = attention_size
        #0: no attention
        #1: attention for each token in the targets
        print('[Info] Initialize attention with attention type =', attention_type)
        if attention_type == 1:
            self.context_linear = nn.Linear(lstm_dim * 4, attention_size).to(NetworkConfig.DEVICE)
            self.attention_U = nn.Parameter(torch.randn(1, attention_size))
            self.attention_linear = nn.Linear(lstm_dim * 2, len(self.polar2id)).to(NetworkConfig.DEVICE)

    def build_linear_layers(self):
        self.e2id = {'B':0, 'M':1, 'E':2, 'S':3, 'O':4}
        self.polar2id = {'+':0, '0':1, '-':2}

        self.e_linear = nn.Linear(self.linear_dim, len(self.e2id)).to(NetworkConfig.DEVICE)
        self.sent_linear = nn.Linear(self.linear_dim, len(self.polar2id)).to(NetworkConfig.DEVICE) # +, 0, -
        self.zero = torch.tensor(0.0).to(NetworkConfig.DEVICE)

    def build_attention(self, lstm_outputs):
        if self.attention_type == 1:
            size, hidden_size = lstm_outputs.size()
            sents = []
            for pos in range(size):
                curr_hidden = lstm_outputs[pos]  # hidden_size
                curr_hidden = curr_hidden.view(1, hidden_size).expand(size, hidden_size)  # size * hidden_size
                context = torch.cat([curr_hidden, lstm_outputs], 1)
                context = self.context_linear(context) # size * attention_size
                context = F.relu(context).transpose(0, 1) # attention_size * size
                beta = self.attention_U @ context # 1 * size  #TODO: correct the way to multipty U with each element in context
                beta = beta.squeeze(0) # size
                alpha = F.softmax(beta, 0)  #size
                alpha = alpha.view(size, 1).expand(size, hidden_size) #
                sent = alpha * lstm_outputs  # size * hidden_size #TODO: correct the way to multipy alpha[i] by lstm_outputs[i]
                sent = torch.sum(sent, 0)  # hidden_size
                sent = self.attention_linear(sent)
                sents.append(sent)

            sents = torch.stack(sents, 0)  # size * label_size

            return sents
        else:
            return None


    def build_nn_graph(self, instance):
        lstm_outputs = self.build_features(instance) #sent_len * hidden_size
        nn_output_e_linear = self.e_linear(lstm_outputs)
        nn_output_sent_linear = self.sent_linear(lstm_outputs)
        nn_attention = self.build_attention(lstm_outputs)
        nn_output = (nn_output_e_linear, nn_output_sent_linear, nn_attention)
        return nn_output

    def get_nn_score(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        pos, node_type, label_id = parent_arr

        if node_type != NodeType.scope.value:  # Start, End
            return self.zero #torch.tensor(0.0).to(NetworkConfig.DEVICE)
        else:
            nn_output_e_linear, nn_output_sent_linear, nn_attention = network.nn_output
            label = self.labels[label_id]
            scope_tag = label[:2]
            polar_tag = label[2]
            polar_tag_id = self.polar2id[polar_tag]
            if scope_tag[0] == 'e':
                target_score = nn_output_e_linear[pos][self.e2id[scope_tag[1]]]

                sent_score = 0 #nn_output_sent_linear[pos][polar_tag_id]

                if self.attention_type == 1:
                    sent_score = sent_score + nn_attention[pos][polar_tag_id]


                return target_score + sent_score
            else: #B A
                if scope_tag == 'Be':
                    return self.zero #torch.tensor(0.0).to(NetworkConfig.DEVICE)
                else:
                    target_score = 0
                    if self.attention_type == 1:
                        target_score = target_score + nn_output_e_linear[pos][self.e2id['O']]
                    sent_score = nn_output_sent_linear[pos][polar_tag_id]
                    return target_score + sent_score



class TSReader():
    Stats = {'MAX_WORD_LENGTH':0}

    @staticmethod
    def read_insts(file, is_labeled, number):
        print('Reading file from ', file)
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
                word = fields[0]
                word = re.sub('\d', '0', word)

                FLAGS = re.MULTILINE | re.DOTALL
                def re_sub(pattern, repl):
                    return re.sub(pattern, repl, word, flags=FLAGS)

                word = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
                word = re_sub(r"@\w+", "<user>")

                TSReader.Stats['MAX_WORD_LENGTH'] = max(TSReader.Stats['MAX_WORD_LENGTH'], len(word))
                # ner = fields[1]
                # sent = fields[-1]
                #
                # if ner == 'O':
                #     output = 'O'
                # else:
                #     ner_tag, ner_type = ner.split('-')
                #
                #     if ner_type not in volitionnal_entity_type:
                #         output = 'O'
                #     else:
                #         if sent == '_':
                #             sent = 'neutral'  #TODO: shall we remove it?
                #
                #         sent_symbol = polar2symbol[sent]
                #
                #         output = ner_tag + sent_symbol
                tag = fields[-1]
                postag = fields[-2]
                THER_SENT_3 = fields[-4]
                SENT =  fields[-5] #if fields[-5] == "_" else fields[-5].split(':')[1]
                browncluster5 = fields[2]
                jerbo = fields[3]
                browncluster3 = fields[4]

                ner_tag = tag[0]
                if ner_tag == 'O':
                    output = ner_tag
                else:
                    sent = tag[2:]
                    try:
                        output = ner_tag + polar2symbol[sent]
                    except:
                        print()



                inputs.append((word, postag, SENT, THER_SENT_3, browncluster5, jerbo, browncluster3))
                outputs.append(output)

        f.close()

        return insts


class TScore(Score):
    def __init__(self, compare_type = 0, accumulated = False):
        self.target_fscore = FScore(0.0, 0.0, 0.0)
        self.sentiment_fscore = FScore(0.0, 0.0, 0.0)
        self.compare_type = compare_type
        self.accumulated = accumulated
        if accumulated:
            self.fold_rets = []

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
        if self.compare_type == 0:
            return self.target_fscore.fscore > obj.target_fscore.fscore and self.sentiment_fscore.fscore > obj.sentiment_fscore.fscore
        elif self.compare_type == 1:
            sentiment_fscore_diff = self.sentiment_fscore.fscore - obj.sentiment_fscore.fscore
            return self.target_fscore.fscore > obj.target_fscore.fscore if math.fabs(sentiment_fscore_diff) < 1e-8 else sentiment_fscore_diff > 0
        else: #self.compare_type == 1:
            target_score_diff = self.target_fscore.fscore - obj.target_fscore.fscore
            return self.sentiment_fscore.fscore > obj.sentiment_fscore.fscore if math.fabs(target_score_diff) < 1e-8 else target_score_diff > 0

    def update_score(self, obj):
        self.target_fscore.update_score(obj.target_fscore)
        self.sentiment_fscore.update_score(obj.sentiment_fscore)

    # def __add__(self, other):
    #     self.target_fscore = self.target_fscore + other.target_fscore
    #     self.sentiment_fscore = self.sentiment_fscore + other.sentiment_fscore
    #     return self

    def accumulate(self, obj):
        self.fold_rets.append(obj)
        # self.target_fscore = self.target_fscore + obj.target_fscore
        # self.sentiment_fscore = self.sentiment_fscore + obj.sentiment_fscore

    def get_average(self, indices = None):

        if indices == None:
            indices = list(range(len(self.fold_rets)))
        num_fold = len(indices) + 0.0

        average_score = TScore()
        for idx in indices:
            tscore = self.fold_rets[idx]
            average_score.target_fscore = average_score.target_fscore + tscore.target_fscore
            average_score.sentiment_fscore = average_score.sentiment_fscore + tscore.sentiment_fscore

        average_score.target_fscore = average_score.target_fscore.divide(num_fold)
        average_score.sentiment_fscore = average_score.sentiment_fscore.divide(num_fold)
        return average_score


class sentimentscope_eval(Eval):

    def __init__(self, compare_type = 0):
        super().__init__()
        self.result_path_prefix = 'tmp/result'
        self.compare_type = compare_type

    def set_result_path_prefix(self, path_prefix):
        self.result_path_prefix = path_prefix


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

        result_target_file = self.result_path_prefix + '_target.txt' #'tmp/result_target.txt'
        result_sent_file = self.result_path_prefix +  '_sent.txt' #tmp/result_sent.txt'

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

            words = [word for word, _, _, _, _, _ ,_ in input]

            target_lines = list(zip(words, target_gold, target_pred))
            sent_lines = list(zip(words, sent_gold, sent_pred))

            inst_target_str = '\n'.join(['\t'.join(item) for item in target_lines])
            inst_sent_str = '\n'.join(['\t'.join(item) for item in sent_lines])

            f_target.write(inst_target_str + '\n\n')
            f_sent.write(inst_sent_str + '\n\n')

        f_target.close()
        f_sent.close()


        target_ret = self.eval_by_script(result_target_file)
        sent_ret = self.eval_by_script(result_sent_file)

        ts_score = TScore(compare_type=self.compare_type)
        ts_score.set_scores(target_ret, sent_ret)

        return ts_score


if __name__ == "__main__":

    import random
    import os
    from random import shuffle

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    if not os.path.exists('result'):
        os.mkdir('result')

    # train_file = "data/ts/train.1.conll.train_test"
    # dev_file = "data/ts/test.1.conll.train_test"

    class TSNeuralBuilderType(Enum):
        VANILLA = 0
        ATT = 1

    import argparse
    parser = argparse.ArgumentParser()
    #subparsers = parser.add_subparsers()

    # subparser = subparsers.add_parser("train")
    # subparser.set_defaults(callback=run_train)
    # for arg in dynet_args:
    #     subparser.add_argument(arg)
    parser.add_argument("--numpy-seed", type=int, default=9997)
    parser.add_argument("--lang", choices=["en", "es"], default='en') #
    parser.add_argument("--tag-embedding-dim", type=int, default=50)
    parser.add_argument("--word-embedding-dim", type=int, default=100)
    parser.add_argument("--char-embedding-dim", type=int, default=50)
    parser.add_argument("--attention-type", type=int, default=0)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dim", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--trial", action="store_true")
    #parser.add_argument("--model-path-base", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--compare-type", type=int, default=0)
    parser.add_argument("--check-every", type=int, default=4000)
    parser.add_argument("--print-vocabs", action="store_true")
    parser.add_argument("--modelname", default="ss")
    parser.add_argument("--neuraltype", default="VANILLA")

    args = parser.parse_args()

    seed = args.numpy_seed
    lang = args.lang
    TRIAL = args.trial
    fold_start_idx = 1
    fold_end_idx = 10 if not TRIAL else 1
    num_train = -1
    num_dev = -1
    num_test = -1
    num_iter = args.epochs
    batch_size = args.batch_size
    num_thread = 1
    NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH = True
    NetworkConfig.GPU_ID = args.gpuid
    emb = ('embedding/glove.6B.100d.txt', 100) if lang == 'en' else ('embedding/SBW-vectors-300-min5.txt', 300)
    embed_path = emb[0]
    word_embed_dim = emb[1]
    postag_embed_dim = 0 # args.tag_embedding_dim
    char_embed_dim = args.char_embedding_dim
    SENT_embed_dim = 0
    THER_SENT_embed_dim = 0
    lstm_dim = args.lstm_dim #word_embed_dim + char_embed_dim + 100
    check_every = args.check_every #700 if lang == 'en' else 1500
    SEPARATE_DEV_FROM_TRAIN = True
    APPEND_START_END = False
    NetworkConfig.ECHO_TEST_RESULT_DURING_EVAL_ON_DEV = True
    visual = False
    DEBUG = False
    neural_builder_type = TSNeuralBuilderType[args.neuraltype]
    attention_type = args.attention_type
    attention_size = lstm_dim
    compare_type = args.compare_type
    model_name = args.modelname + '_' + neural_builder_type.name

    print(args)



    overall_score = TScore(accumulated=True)
    for num_fold in range(fold_start_idx, fold_end_idx + 1):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(40)

        print(colored('Fold ', 'blue'), num_fold, '')

        num_fold_str = str(num_fold)
        path = 'data/ts/' + lang + '/'
        train_file = path + "train." + num_fold_str + ".coll"
        dev_file = path + "test." + num_fold_str + ".coll"
        test_file = path + "test." + num_fold_str + ".coll"
        dev_file = test_file
        trial_file = "data/ts/trial.txt"
        model_path = 'result/' + model_name + '_' + lang + '_' + num_fold_str + ".pt"


        if TRIAL == True:
            data_size = -1
            train_file = trial_file
            dev_file = trial_file
            test_file = trial_file
            num_iter = 30
            check_every = 4
            NetworkConfig.GPU_ID = -1
            embed_path = None
            SEPARATE_DEV_FROM_TRAIN = False
            # if num_fold > 1:
            #     break

        if NetworkConfig.GPU_ID > -1:
            NetworkConfig.DEVICE = torch.device("cuda:" + str(NetworkConfig.GPU_ID))

        if num_thread > 1:
            NetworkConfig.NUM_THREADS = num_thread
            print('Set NUM_THREADS = ', num_thread)

        train_insts = TSReader.read_insts(train_file, True, num_train)
        dev_insts = TSReader.read_insts(dev_file, False, num_dev)
        test_insts = TSReader.read_insts(test_file, False, num_test)


        vocab2id = {PAD:0, UNK:1}
        postag2id = {PAD:0, UNK:1}
        char2id = {PAD:0, UNK:1}
        SENT2id = {PAD:0, UNK:1}
        THER_SENT2id = {PAD:0, UNK:1}

        browncluster52id = {PAD: 0, UNK: 1}
        browncluster32id = {PAD: 0, UNK: 1}
        jerbo2id = {PAD: 0, UNK: 1}

        label2id = {}
        scope_tags = ['BB', 'Be', 'eB', 'eM', 'eE', 'eS', 'AA', 'AB+', 'AB0', 'AB-']
        polar_tags = ['+', '0', '-']

        for scope_tag in scope_tags:
            for polar in polar_tags:
                label2id[scope_tag + polar] = len(label2id)


        labels = [None] * (len(label2id))
        for key in label2id:
            labels[label2id[key]] = key

        if DEBUG:
            print('label2id:',list(label2id.keys()))
            print('postag2id:', list(postag2id.keys()))


        for inst in train_insts:  #+ dev_insts + test_insts:
            for word, postag, SENT, THER_SENT_3, browncluster5, jerbo, browncluster3 in inst.input:
                if word not in vocab2id:
                    vocab2id[word] = len(vocab2id)

                    for ch in word:
                        if ch not in char2id:
                            char2id[ch] = len(char2id)

                if postag not in postag2id:
                    postag2id[postag] = len(postag2id)

                if SENT not in SENT2id:
                    SENT2id[SENT] = len(SENT2id)


                if THER_SENT_3 not in THER_SENT2id:
                    THER_SENT2id[THER_SENT_3] = len(THER_SENT2id)

                if browncluster5 not in browncluster52id:
                    browncluster52id[browncluster5] = len(browncluster52id)


                if browncluster3 not in browncluster32id:
                    browncluster32id[browncluster3] = len(browncluster32id)

                if jerbo not in jerbo2id:
                    jerbo2id[jerbo] = len(jerbo2id)



        SENT_embed_dim = len(SENT2id)
        THER_SENT_embed_dim = len(THER_SENT2id)
        #postag_embed_dim = len(postag2id)
        browncluster5_embed_dim = len(browncluster52id)
        browncluster3_embed_dim = len(browncluster32id)


        chars = [None] * len(char2id)
        for key in char2id:
            chars[char2id[key]] = key

        max_word_length = TSReader.Stats['MAX_WORD_LENGTH']
        print(colored('MAX_WORD_LENGTH:', 'blue'), TSReader.Stats['MAX_WORD_LENGTH'])


        for inst in train_insts: # + dev_insts + test_insts:
            input_seq = inst.input
            inst.word_seq = torch.tensor([vocab2id[word] for word, _ , _ , _, _, _, _ in input_seq]).to(NetworkConfig.DEVICE)
            inst.postag_seq = torch.tensor([postag2id[postag] for _, postag, _, _, _, _, _ in input_seq]).to(NetworkConfig.DEVICE)
            char_seq_list = [[char2id[ch] for ch in word] + [char2id[PAD]] * (max_word_length - len(word)) for word, _ , _ , _, _, _, _  in input_seq]
            inst.char_seq_tensor = torch.tensor(char_seq_list).to(NetworkConfig.DEVICE)
            # char_seq_tensor: (1, sent_len, word_length)
            inst.char_seq_len = torch.tensor([len(word) for word, _ , _ , _, _, _, _ in input_seq]).to(NetworkConfig.DEVICE)
            # char_seq_len: (1, sent_len)
            inst.SENT_seq = torch.tensor([SENT2id[SENT] for _, _, SENT, _ ,_, _, _ in input_seq]).to(NetworkConfig.DEVICE)
            inst.THER_SENT_seq = torch.tensor([THER_SENT2id[THER_SENT_3] for _, _, _,  THER_SENT_3, _, _, _ in input_seq]).to(NetworkConfig.DEVICE)

            inst.browncluster5_seq = torch.tensor([browncluster52id[browncluster5] for _, _, _, _, browncluster5, _, _ in input_seq]).to(NetworkConfig.DEVICE)
            inst.browncluster3_seq = torch.tensor([browncluster32id[browncluster3] for _, _, _, _, _, _, browncluster3 in input_seq]).to(NetworkConfig.DEVICE)

        if SEPARATE_DEV_FROM_TRAIN:
            old_train_insts = list(train_insts)
            shuffle(old_train_insts)
            train_size = len(old_train_insts)
            train_dev_split = (train_size // 10) * 9
            train_insts = old_train_insts[:train_dev_split]
            dev_insts = old_train_insts[train_dev_split:]
            for inst in dev_insts:
                inst.is_labeled = False


        for inst in dev_insts + test_insts:
            input_seq = inst.input
            inst.word_seq = torch.tensor([vocab2id[word] if word in vocab2id else vocab2id[UNK] for word, _, _, _, _, _ ,_ in input_seq]).to(NetworkConfig.DEVICE)
            inst.postag_seq = torch.tensor([postag2id[postag] if word in postag2id else postag2id[UNK] for _, postag, _, _, _, _ ,_ in input_seq]).to(NetworkConfig.DEVICE)
            char_seq_list = [[char2id[ch] if ch in char2id else char2id[UNK] for ch in word] + [char2id[PAD]] * (max_word_length - len(word)) for word, _, _, _, _, _ ,_ in input_seq]
            inst.char_seq_tensor = torch.tensor(char_seq_list).to(NetworkConfig.DEVICE)
            # char_seq_tensor: (1, sent_len, word_length)
            inst.char_seq_len = torch.tensor([len(word) for word, _, _, _, _, _ ,_ in input_seq]).to(NetworkConfig.DEVICE)
            inst.SENT_seq = torch.tensor([SENT2id[SENT] if SENT in SENT2id else SENT2id[UNK] for _, _, SENT, _, _, _ ,_ in input_seq]).to(NetworkConfig.DEVICE)
            inst.THER_SENT_seq = torch.tensor([THER_SENT2id[THER_SENT_3] if THER_SENT_3 in THER_SENT2id else THER_SENT2id[UNK] for _, _, _, THER_SENT_3, _, _ ,_ in input_seq]).to(NetworkConfig.DEVICE)
            # char_seq_len: (1, sent_len)

            inst.browncluster5_seq = torch.tensor([browncluster52id[browncluster5] if browncluster5 in browncluster52id else browncluster52id[UNK] for _, _, _, _, browncluster5, _, _ in input_seq]).to(NetworkConfig.DEVICE)
            inst.browncluster3_seq = torch.tensor([browncluster32id[browncluster3] if browncluster3 in browncluster32id else browncluster32id[UNK] for _, _, _, _, _, _, browncluster3 in input_seq]).to(NetworkConfig.DEVICE)

        print('Train:', len(train_insts), '\tDev:', len(dev_insts), '\tTest:', len(test_insts))

        gnp = TensorGlobalNetworkParam()
        if neural_builder_type == TSNeuralBuilderType.VANILLA:
            neural_builder =    TSNeuralBuilder(gnp, labels, len(vocab2id), word_embed_dim, len(postag2id), postag_embed_dim, char_embed_dim, char_embed_dim, len(SENT2id) ,SENT_embed_dim, THER_SENT_embed_dim, browncluster5_embed_dim, browncluster3_embed_dim, lstm_dim = lstm_dim, dropout=args.dropout)
        elif neural_builder_type == TSNeuralBuilderType.ATT:
            neural_builder = TSATTNeuralBuilder(gnp, labels, len(vocab2id), word_embed_dim, len(postag2id), postag_embed_dim, char_embed_dim, char_embed_dim, len(SENT2id), SENT_embed_dim, THER_SENT_embed_dim, browncluster5_embed_dim, browncluster3_embed_dim, lstm_dim = lstm_dim, dropout=args.dropout)
            neural_builder.init_attention(attention_type, attention_size)
        else:#gnp, labels, voc_size, word_embed_dim, postag_size, postag_embed_dim, char_emb_size, charlstm_hidden_dim, SENT_emb_size, SENT_embed_dim, browncluster5_embed_dim, browncluster3_embed_dim, lstm_dim, dropout)
            print('Unsupported TS Neural Builder Type')
            exit()

        neural_builder.load_pretrain(vocab2id, embed_path)
        compiler = TSNetworkCompiler(label2id, labels, scope_tags, polar_tags)
        evaluator = sentimentscope_eval(compare_type=compare_type)
        model = NetworkModel(neural_builder, compiler, evaluator)
        model.model_path = model_path
        model.check_every = check_every

        if visual:
            from hypergraph.Visualizer import Visualizer
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
                            return self.input[node_arr[0]][0]
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
            ts_visualizer = TSVisualizer(compiler, neural_builder, labels)
            inst = train_insts[0]
            inst.is_labeled = True
            ts_visualizer.visualize_inst(inst)
            exit()

        if batch_size == 1:
            model.learn(train_insts, num_iter, dev_insts, test_insts)
        else:
            model.learn_batch(train_insts, num_iter, dev_insts, batch_size)

        #try:
        model.load()

        results = model.test(test_insts)
        if TRIAL:
            for inst in results:
                print(inst.get_input())
                print(inst.get_output())
                print(inst.get_prediction())
                print()
        evaluator.set_result_path_prefix('result/' + model_name +'_' + num_fold_str)
        ret = model.evaluator.eval(results)
        print(ret)


        overall_score.accumulate(ret)
        # except:
        #     print('Loading model fails ...')
        #     exit()

        print()
        print()


    average_score = overall_score.get_average()
    print(colored('Overall Result===============', 'blue'))
    print('Result across folds ', fold_start_idx, ' .. ', fold_end_idx)
    print(average_score)
    print()
    if not TRIAL:
        average_score = overall_score.get_average(indices=range(1, fold_end_idx))
        print('Result across 9 folds ', 2, ' .. ', fold_end_idx)
        print(average_score)
        print()
        print('Results of each folds:')
        for i in range(len(overall_score.fold_rets)):
            print(overall_score.fold_rets[i])



