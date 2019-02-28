from hypergraph.Utils import Eval
from PYEVALB import scorer
import re
import math
import os
import subprocess
import examples.parsingtree.trees as trees

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
    def __init__(self, precision, recall, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(Precision={:.2f}%, Recall={:.2f}%, FScore={:.2f}%)".format(
            self.precision * 100, self.recall * 100, self.fscore * 100)


    def to_tuple(self):
        return [self.precision, self.recall, self.fscore]


    def larger_than(self, obj):
        return self.fscore > obj.fscore

    def update_score(self, obj):
        self.recall = obj.recall
        self.precision = obj.precision
        self.fscore = obj.fscore

    def __add__(self, other):
        self.recall += other.recall
        self.precision += other.precision
        self.fscore += other.fscore
        return self

    def divide(self, n):
        self.recall /= n
        self.precision /= n
        self.fscore /= n
        return self

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

        precision = p * 1.0 / total_predict  if total_predict != 0 else 0
        recall = p * 1.0 / total_entity  if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

        #ret = [precision, recall, fscore]
        fscore = FScore(recall, precision, fscore)

        return fscore


class semieval(Eval):

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
                if output[i][2] != "O":
                    output_spans.add(Span(output[i][0], output[i][1], output[i][2]))
            predict_spans = set()
            for i in range(len(prediction)):
                if prediction[i][2] != "O":
                    predict_spans.add(Span(prediction[i][0], prediction[i][1], prediction[i][2]))

            total_entity += len(output_spans)
            total_predict += len(predict_spans)
            p += len(predict_spans.intersection(output_spans))

        precision = p * 1.0 / total_predict  if total_predict != 0 else 0
        recall = p * 1.0 / total_entity  if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

        #ret = [precision, recall, fscore]
        fscore = FScore(recall, precision, fscore)

        return fscore


class constituent_eval(Eval):

    def evalb(self, evalb_dir, gold_trees, predicted_trees):

        assert os.path.exists(evalb_dir)
        evalb_program_path = os.path.join(evalb_dir, "evalb")
        evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
        assert os.path.exists(evalb_program_path)
        assert os.path.exists(evalb_param_path)

        assert len(gold_trees) == len(predicted_trees)
        for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
            assert isinstance(gold_tree, trees.TreebankNode)
            assert isinstance(predicted_tree, trees.TreebankNode)
            gold_leaves = list(gold_tree.leaves())
            predicted_leaves = list(predicted_tree.leaves())
            assert len(gold_leaves) == len(predicted_leaves)
            assert all(
                gold_leaf.word == predicted_leaf.word
                for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves))

        temp_dir = "tmp"  # tempfile.TemporaryDirectory(prefix="evalb-")
        gold_path = os.path.join(temp_dir, "gold.txt")
        predicted_path = os.path.join(temp_dir, "predicted.txt")
        output_path = os.path.join(temp_dir, "output.txt")

        with open(gold_path, "w") as outfile:
            for tree in gold_trees:
                outfile.write("{}\n".format(tree.linearize()))

        with open(predicted_path, "w") as outfile:
            for tree in predicted_trees:
                outfile.write("{}\n".format(tree.linearize()))

        command = "{} -p {} {} {} > {}".format(
            evalb_program_path,
            evalb_param_path,
            gold_path,
            predicted_path,
            output_path,
        )
        subprocess.run(command, shell=True)

        fscore = FScore(math.nan, math.nan, math.nan)
        with open(output_path) as infile:
            for line in infile:
                match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
                if match:
                    fscore.recall = float(match.group(1)) / 100
                match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
                if match:
                    fscore.precision = float(match.group(1)) / 100
                match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
                if match:
                    fscore.fscore = float(match.group(1)) / 100
                    break

        success = (
                not math.isnan(fscore.fscore) or
                fscore.recall == 0.0 or
                fscore.precision == 0.0)

        if success:
            pass  # temp_dir.cleanup()
        else:
            print("Error reading EVALB results.")
            print("Gold path: {}".format(gold_path))
            print("Predicted path: {}".format(predicted_path))
            print("Output path: {}".format(output_path))

        return fscore

    def eval(self, insts):

        gold_path = 'tmp/gold.txt'
        pred_path = 'tmp/pred.txt'
        result_path = 'tmp/result.txt'

        fgold = open(gold_path, 'w', encoding='utf-8')
        fpred = open(pred_path, 'w', encoding='utf-8')
        golds = []
        preds = []
        for inst in insts:
            gold = inst.get_output()
            pred = inst.get_prediction()

            golds.append(gold)
            preds.append(pred)

            fgold.write(gold.linearize() + '\n')
            fpred.write(pred.linearize() + '\n')

        fgold.close()
        fpred.close()

        return self.evalb('./EVALB', golds, preds)

        evalb = scorer.Scorer()

        fscore = FScore(0.0, 0.0, 0.0)

        try:
            evalb.evalb(gold_path, pred_path, result_path)
            with open(result_path) as infile:
                for line in infile:
                    match = re.match(r"Bracketing Recall:\s+(\d+\.\d+)", line)
                    if match:
                        fscore.recall = float(match.group(1)) / 100
                    match = re.match(r"Bracketing Precision:\s+(\d+\.\d+)", line)
                    if match:
                        fscore.precision = float(match.group(1)) / 100
                    match = re.match(r"Bracketing FMeasure:\s+(\d+\.\d+)", line)
                    if match:
                        fscore.fscore = float(match.group(1)) / 100
                        break

        except:
            pass



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

        return fscore


class label_eval(Eval):
    def eval(self, insts):

        p = 0

        for inst in insts:

            output = inst.output
            prediction = inst.prediction
            #convert to span
            if output == prediction:
                p += 1


        #ret = [precision, recall, fscore]
        acc = p / len(insts)
        fscore = FScore(acc, acc, acc)

        return fscore