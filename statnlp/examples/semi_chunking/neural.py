from statnlp.hypergraph.NeuralBuilder import NeuralBuilder
from statnlp.hypergraph.Utils import *
import torch.nn as nn

class SemiNeural(NeuralBuilder):

    def __init__(self, gnp, voc_size, label_size, char2id, chars, char_emb_size, charlstm_hidden_dim, lstm_hidden_size = 100, dropout = 0.5):
        super().__init__(gnp)
        self.token_embed = 100
        self.label_size = label_size
        print("vocab size: ", voc_size)
        # self.word_embed = nn.Embedding(voc_size, self.token_embed, padding_idx=0).to(NetworkConfig.DEVICE)
        self.word_embed = nn.Embedding(voc_size, self.token_embed).to(NetworkConfig.DEVICE)

        self.lstm_dim = lstm_hidden_size
        self.char_emb_size = char_emb_size
        if char_emb_size > 0:
            from statnlp.features.char_lstm import CharBiLSTM
            self.char_bilstm = CharBiLSTM(char2id, chars, char_emb_size, charlstm_hidden_dim).to(NetworkConfig.DEVICE)
            # self.char_embeddings = nn.Embedding(len(char2id), self.char_emb_size).to(NetworkConfig.DEVICE)
            # self.char_rnn = nn.LSTM(self.char_emb_size, charlstm_hidden_dim//2, batch_first=True, bidirectional=True).to(NetworkConfig.DEVICE)
        self.word_drop = nn.Dropout(dropout).to(NetworkConfig.DEVICE)


        lstm_input_size = self.token_embed + charlstm_hidden_dim

        self.rnn = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True,bidirectional=True).to(NetworkConfig.DEVICE)
        self.f_label = nn.Linear(lstm_hidden_size * 2, label_size).to(NetworkConfig.DEVICE)


        self.init = nn.Parameter(torch.randn(lstm_hidden_size ))
        self.last = nn.Parameter(torch.randn(lstm_hidden_size ))

        # label_hidden_size = 100
        # self.f_label = nn.Sequential(
            # nn.Linear(lstm_hidden_size * 2, label_hidden_size).to(NetworkConfig.DEVICE),
            # nn.ReLU().to(NetworkConfig.DEVICE),
            # nn.Linear(lstm_hidden_size * 2, label_size).to(NetworkConfig.DEVICE)
        # ).to(NetworkConfig.DEVICE)

    def load_pretrain(self, path, word2idx):
        emb = load_emb_glove(path, word2idx, self.token_embed)
        self.word_embed.weight.data.copy_(torch.from_numpy(emb))
        self.word_embed = self.word_embed.to(NetworkConfig.DEVICE)

    # @abstractmethod
    # def extract_helper(self, network, parent_k, children_k, children_k_index):
    #     pass
    def build_nn_graph(self, instance):

        word_seq = instance.word_seq
        word_vec = self.word_embed(word_seq).unsqueeze(0) ###1 x sent_len x embedding size.
        word_rep = [word_vec]

        if self.char_emb_size > 0:
            char_seq_tensor = instance.char_seq_tensor.unsqueeze(0)
            char_seq_len = instance.char_seq_len.unsqueeze(0)
            char_features = self.char_bilstm.get_last_hiddens(char_seq_tensor, char_seq_len)  # batch_size, sent_len, char_hidden_dim
            word_rep.append(char_features)

        word_rep = torch.cat(word_rep, 2)
        word_rep = self.word_drop(word_rep)
        #
        lstm_out, _ = self.rnn(word_rep, None)
        lstm_out = lstm_out.squeeze(0) # sent_len * hidden_size

        # linear_output = self.f_label(lstm_out)
        # zero_col = torch.zeros(1, self.label_size).to(NetworkConfig.DEVICE)
        # return torch.cat([linear_output, zero_col], 0)
        sent_len, lstm_dim = lstm_out.size()

        square = lstm_out.view(sent_len, 1, lstm_dim).expand(sent_len, sent_len, lstm_dim)
        square_t = square.transpose(0, 1)

        ret = square_t - square
        ret = ret[:, 1: sent_len - 1, :]

        half_lstm_dim = lstm_dim // 2

        fwd = ret[:sent_len - 2, :, :half_lstm_dim]

        bwd = ret[2:, :, half_lstm_dim:].transpose(0, 1)

        bi = torch.cat([fwd, bwd], 2)  ## sent_len x sent_len x hidden size
        spans = self.f_label(bi) ## sent_len x sent_len x num_labels

        zeros = torch.zeros((sent_len - 2, sent_len - 2, 1)).to(NetworkConfig.DEVICE)  # score of (), empty label
        spans = torch.cat([zeros, spans], 2)

        spans[sent_len - 3, 0, 0] = 0
        return spans

    def get_nn_score(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)  # pos, label_id, node_type
        right, length, label_id, node_type = parent_arr
        left = right - length + 1

        if node_type != 1:  # Start, End
            return torch.tensor(0.0).to(NetworkConfig.DEVICE)
        else:
            spans = network.nn_output
            return spans[left, right][label_id]


    def get_label_id(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        return parent_arr[2]

    def build_node2nn_output(self, network):
        # size = network.count_nodes()
        # sent_len = network.get_instance().size()
        # nodeid2nn = [0] * size
        # for k in range(size):
        #     parent_arr = network.get_node_array(k)  # pos, label_id, node_type
        #     end = parent_arr[0]
        #     label_id = parent_arr[2]
        #     node_type = parent_arr[3]
        #
        #     if node_type != 1:  # Start, End
        #         idx = sent_len * self.label_size
        #     else:
        #         idx = end * self.label_size + label_id
        #     nodeid2nn[k] = idx
        # return nodeid2nn


        num_nodes = network.count_nodes()
        nodeid2nn = [0] * num_nodes
        for k in range(num_nodes):
            parent_arr = network.get_node_array(k)  # pos, label_id, node_type
            size = network.get_instance().size()
            right, length, label_id, node_type = parent_arr
            left = right - length + 1  ## right is inclusive

            if node_type != 1: ### not label node.
                idx = (size - 1) * size * self.label_size ## a index with 0
            else:
                row = left * size + right
                idx = row * self.label_size + label_id
            nodeid2nn[k] = idx
        return nodeid2nn


