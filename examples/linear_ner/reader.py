
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
            if is_train:
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