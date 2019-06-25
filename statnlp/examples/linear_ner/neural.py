from statnlp.hypergraph.NeuralBuilder import NeuralBuilder
from statnlp.hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from statnlp.hypergraph.Utils import *
from statnlp.features.char_lstm import CharBiLSTM ## we have implemented this and you can use it directly.
import torch.nn as nn

class LSTMBuilder(NeuralBuilder):
    def __init__(self, gnp: TensorGlobalNetworkParam, vocab_size, label_size, numpy_word_embedding, char2id, chars, device):
        super().__init__(gnp)
        self.token_embed = 100
        self.label_size = label_size
        print("vocab size: ", vocab_size)
        self.char_emb = 25
        self.char_hidden = 50
        self.dropout = 0.5
        lstm_input_size = self.token_embed
        self.lstm_hidden_size = 200
        self.device = device
        if self.char_emb > 0:
            self.char_bilstm = CharBiLSTM(char2id, chars, self.char_emb, self.char_hidden).to(device)
            lstm_input_size += self.char_hidden
        self.word_drop = nn.Dropout(self.dropout).to(device)
        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_word_embedding)).to(device)
        self.lstm_drop = nn.Dropout(self.dropout).to(device)
        self.rnn = nn.LSTM(lstm_input_size, self.lstm_hidden_size, batch_first=True,bidirectional=True).to(device)
        self.linear = nn.Linear(self.lstm_hidden_size * 2, self.label_size).to(device)


    def build_nn_graph(self, instance):

        word_rep = self.word_embed(instance.word_seq).unsqueeze(0) ###1 x sent_len x embedding size.
        if self.char_emb > 0:
            char_seq_tensor = instance.char_seq_tensor.unsqueeze(0)
            char_seq_len = instance.char_seq_len.unsqueeze(0)
            char_features = self.char_bilstm.get_last_hiddens(char_seq_tensor, char_seq_len)  # batch_size, sent_len, char_hidden_dim
            word_rep = [word_rep, char_features]
            word_rep = torch.cat(word_rep, 2)
        word_rep = self.word_drop(word_rep)
        lstm_out, _ = self.rnn(word_rep, None)
        lstm_out = self.lstm_drop(lstm_out)
        linear_output = self.linear(lstm_out).squeeze(0)
        zero_col = torch.zeros(1, self.label_size).to(self.device) ## zero_col is used to obtain elements with value 0
        return torch.cat([linear_output, zero_col], 0) ## The output has size: (sent_len+1, number of labels)

    def get_label_id(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        return parent_arr[1]

    def build_node2nn_output(self, network):
        size = network.count_nodes()
        sent_len = network.get_instance().size()
        nodeid2nn = [0] * size
        for k in range(size):
            parent_arr = network.get_node_array(k)  # pos, label_id, node_type
            pos = parent_arr[0]
            label_id = parent_arr[1]
            node_type = parent_arr[2]

            if node_type == 0 or node_type == 2:  # Start or End
                idx = sent_len * self.label_size
            else:
                idx = pos * self.label_size  + label_id
            nodeid2nn[k] = idx
        return nodeid2nn

