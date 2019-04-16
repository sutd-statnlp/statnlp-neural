# Tutorial Basics: Load data into instances

In this tutorial, we are going to build a __linear-chain CRF__ model for named entity recognition using StatNLP.

## Instance

The `Instance` class is the basic component during the inference in StatNLP. We have pre-defined several subclasses that inherit this class. For example, `LinearInstance`:

```python
from hypergraph.Instance import Instance
class LinearInstance(Instance):
    def __init__(self, instance_id, weight, input, output):
        super().__init__(instance_id, weight, input, output)
        self.word_seq = None

    def size(self):
        # print('input:', self.input)
        return len(self.input)

    ...
```

where the `input` should be a sentence and `output` should be the label sequence. The `word_seq` here is used to represent the input word sequence denoted by word ids. __In this tutorial, you do not have to worry about implementing a new `Instance` class__.

## Read the data
As we are focusing on the task of named entity recognition, our datasets contain three columns where the __first column is the word__, and the __third column is the named entity label__. The following code is under `examples/linear_ner/reader.py`.

```python
from common.LinearInstance import LinearInstance
import re
def read_insts(file, is_train, number):
    insts = []
    inputs = []
    outputs = []
    f = open(file, 'r', encoding='utf-8')
    for line in f:
        line = line.strip()
        if len(line) == 0:
            inst = LinearInstance(len(insts) + 1, 1, inputs, outputs)
            if is_train:  ## this part is import tant.
                inst.set_labeled()
            else:
                inst.set_unlabeled()
            insts.append(inst)
            inputs = []
            outputs = []
            if len(insts) >= number and number > 0:
                break
        else:
            fields = line.split()
            input = fields[0]
            input = re.sub('\d', '0', input)
            output = fields[-1]
            inputs.append(input)
            outputs.append(output)
    f.close()
    return insts
```
The argument `is_train` is important to determine whether we are going to use the function `set_labeled()` or `set_unlabeled()`. If `is_train` is `True`, we set the instance to be labeled, indicating we are going to build a __labeled network__ for this instance.
