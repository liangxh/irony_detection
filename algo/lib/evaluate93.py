# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import numpy as np
from algo.model.const import *

NUM_CLASSES = 4
label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


def label_to_one_hot(label, dim=4):
    vec = [0.] * dim
    vec[label] = 1
    return vec


def getMetrics(labels_predict, labels_gold):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]

    discretePredictions = np.asarray(list(map(label_to_one_hot, labels_predict)))
    predictions = discretePredictions
    ground = np.asarray(list(map(label_to_one_hot, labels_gold)))

    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (
                                                                                     macroPrecision + macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
    macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (
                                                                                     microPrecision + microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def test():
    n_sample = 1000
    labels_predict = np.asarray([random.randint(0, 3) for _ in range(n_sample)])
    labels_gold = np.asarray([random.randint(0, 3) for _ in range(n_sample)])

    my_res = basic_evaluate(labels_gold, labels_predict)

    res = getMetrics(labels_predict, labels_gold)
    print()
    print(*res[1:])
    print(microPrecision, microRecall, microF1)


def basic_evaluate(gold, pred):
    n_sample = len(gold)

    predictions = np.asarray(list(map(label_to_one_hot, pred)))
    ground = np.asarray(list(map(label_to_one_hot, gold)))

    match = predictions * ground

    idx = np.asarray(range(n_sample))

    print(match[:, 1:].sum(), predictions[:, 1:].sum())
    print(match[:, 1:].sum(), ground[:, 1:].sum())

    microPrecision = float(match[:, 1:].sum()) / float(predictions[:, 1:].sum())
    microRecall = float(match[:, 1:].sum()) / float(ground[:, 1:].sum())

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (microPrecision + microRecall) > 0 else 0
    accuracy = float(match.sum()) / n_sample

    return {
        ACCURACY: accuracy,
        PRECISION: microPrecision,
        RECALL: microRecall,
        F1_SCORE: microF1
    }


if __name__ == '__main__':
    test()