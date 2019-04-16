# Neural Network: Design Your Neural Net!

This section initially should be talking about the feature manager if it is a Java framework with discrete features. As the features are replaced by the neural networks, this section will be talking about __the design of neural networks__. We have to implement the following compulsory functions:
* Building the neural network: [build_nn_graph(self, instance)](#Forward) function
* [get_label_id(self, network, parent_k)](#Let-StatNLP-knows-what-is-the-label): let statnlp knows the label position of a specific node.
* [build_node2nn_output(self, network)](#Map-Neural-Network-Output-to-Graphical-Model): build the mapping from neural network to the graphical model. 

## Initialize the Neural Network
Similar to what we did in PyTorch, we initialize a model class that inherits `NeuralBuilder` and pass in the hyperparaters (such as hidden size) of a model. In this example, we fixed them to specific numbers for a clean example.

```python
from hypergraph.NeuralBuilder import NeuralBuilder
from hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from hypergraph.Utils import *
from features.char_lstm import CharBiLSTM ## we have implemented this and you can use it directly.
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

        if self.char_emb_size > 0:
            self.char_bilstm = CharBiLSTM(char2id, chars, self.char_emb, self.char_hidden).to(device)
            lstm_input_size += self.charlstm_hidden_dim
        self.word_drop = nn.Dropout(self.dropout).to(device)
        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_word_embedding)).to(device)
        self.lstm_drop = nn.Dropout(self.dropout).to(device)
        self.rnn = nn.LSTM(lstm_input_size, self.lstm_hidden_size, batch_first=True,bidirectional=True).to(device)
        self.linear = nn.Linear(self.lstm_hidden_size * 2, self.label_size).to(device)
```

## Forward
Implement the process on how the model forward. In this example, we forward the instance through LSTM and obtain the scores for each label. __Note that the batch size currently in StatNLP is always 1.__
```python
    def build_nn_graph(self, instance):

        word_rep = self.word_embed(instance.word_seq).unsqueeze(0) ###1 x sent_len x embedding size.
        if self.char_emb_size > 0:
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
        return torch.cat([linear_output, zero_col], 0)   ## The output has size: (sent_len+1, number of labels)
```
The resulting output matrix is as follows with size `sent_len +1 x number of labels`.  The sentence length is 4 and the number of labels is 3 below. 

![alt text](/docs/lstm_output.png)

## Let StatNLP knows what is the label
The following function know the position of the label in the node array defined in the network compiler.
```python
    def get_label_id(self, network, parent_k):
        parent_arr = network.get_node_array(parent_k)
        return parent_arr[1]
```

## Map Neural Network Output to Graphical Model
Finally, we need to build the mapping from the neural network output to each node in the graphical model. We use the figure below to illustrate the mapping process. 
![alt text](/docs/mapping.png)

For example, we have a node `k` with node array `[1,2,1]` under the encoding scheme `[position, label_id, node_type]`. We have the informaiton that the position is `1` and the label id is `2`. Then it should match the `score[1,2]` in the output matrix. To do this, we treat output matrix as a linearized array, and then the index of `score[1,2]` should be `position x label_size + label_id = 1 x 3 + 2 = 5`. Thus, we return the idx `idx=5` for the node `k`, `nodeid2nn[k]=idx`. For some of the nodes that are not used, such as leaf and root, we need to assign the position that have zero score, that is why we concatenate a zero row while building the network. Thus, for leaf and root, the idx should be `idx = 4 x 3 = 12`.
```python
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
```


That's all for the builder! Next, MAIN function!