from hypergraph.Utils import Eval
from PYEVALB import scorer
import re
import math
class Span:

    def __init__(self, left, right, type):
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))


class FScore(object):
    def __init__(self, recall, precision, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(Recall={:.2f}%, Precision={:.2f}%, FScore={:.2f}%)".format(
            self.recall * 100, self.precision * 100, self.fscore * 100)


    def to_tuple(self):
        return [self.precision, self.recall, self.fscore]

## the input to the evaluation should already have
## have the predictions which is the label.
## iobest tagging scheme
class nereval(Eval):
    def eval(self, insts):

        p = 0
        total_entity = 0
        total_predict = 0

        for inst in insts:

            output = inst.output
            prediction = inst.prediction
            #convert to span
            output_spans = set()
            start = -1
            for i in range(len(output)):
                if output[i].startswith("B-"):
                    start = i
                if output[i].startswith("E-"):
                    end = i
                    output_spans.add(Span(start, end, output[i][2:]))
                if output[i].startswith("S-"):
                    output_spans.add(Span(i, i, output[i][2:]))
            predict_spans = set()
            for i in range(len(prediction)):
                if prediction[i].startswith("B-"):
                    start = i
                if prediction[i].startswith("E-"):
                    end = i
                    predict_spans.add(Span(start, end, prediction[i][2:]))
                if prediction[i].startswith("S-"):
                    predict_spans.add(Span(i, i, prediction[i][2:]))

            total_entity += len(output_spans)
            total_predict += len(predict_spans)
            p += len(predict_spans.intersection(output_spans))

        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

        ret = [precision, recall, fscore]


        return ret


class constituent_eval(Eval):
    def eval(self, insts):


        gold_path = '../tmp/gold.txt'
        pred_path = '../tmp/pred.txt'
        result_path = '../tmp/result.txt'

        fgold = open(gold_path, 'w', encoding='utf-8')
        fpred = open(pred_path, 'w', encoding='utf-8')
        for inst in insts:
            gold = inst.get_output()
            pred = inst.get_prediction()

            fgold.write(gold.linearize() + '\n')
            fpred.write(pred.linearize() + '\n')

        fgold.close()
        fpred.close()

        evalb = scorer.Scorer()

        evalb.evalb(gold_path, pred_path, result_path)


        fscore = FScore(0.0, 0.0, 0.0)
        with open(result_path) as infile:
            for line in infile:
                match = re.match(r"Bracketing Recall:\s+(\d+\.\d+)", line)
                if match:
                    fscore.recall = float(match.group(1))
                match = re.match(r"Bracketing Precision:\s+(\d+\.\d+)", line)
                if match:
                    fscore.precision = float(match.group(1))
                match = re.match(r"Bracketing FMeasure:\s+(\d+\.\d+)", line)
                if match:
                    fscore.fscore = float(match.group(1))
                    break

        # success = (
        #         not math.isnan(fscore.fscore) or
        #         fscore.recall == 0.0 or
        #         fscore.precision == 0.0)
        #
        # if success:
        #     pass
        #     # temp_dir.cleanup()
        # else:
        #     print("Error reading EVALB results.")
        #     print("Gold path: {}".format(gold_path))
        #     print("Predicted path: {}".format(pred_path))
        #     print("Output path: {}".format(result_path))

        return fscore.to_tuple()
