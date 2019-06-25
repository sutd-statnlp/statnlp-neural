from statnlp.common import LinearInstance
import re


def read_insts(file, is_labeled, number):
    insts = []
    inputs = []
    outputs = []
    f = open(file, 'r', encoding='utf-8')
    start = 0
    end = 0
    pos = 0
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
            start = 0
            end = 0
            pos = 0
            if len(insts) >= number and number > 0:
                break

        else:
            fields = line.split()
            input = fields[0]
            input = re.sub('\d', '0', input)
            output = fields[-1]


            if output == "O":
                label = "O"
            else:
                label = output[2:]

            if label == "O":
                start = pos
                end = pos
                outputs.append((start, end, label))
            elif output.startswith("E-"):
                end = pos
                outputs.append((start, end, label))
            elif output.startswith("B-"):
                start = pos
            elif output.startswith("S-"):
                start = pos
                end = pos
                outputs.append((start, end, label))
            pos += 1
            inputs.append(input)

    f.close()

    return insts

