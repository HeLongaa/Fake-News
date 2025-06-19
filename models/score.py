# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

LABELS = ['0', '1']


def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0],
          [0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 1
        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-" * line_len)
    lines.append(header)
    lines.append("-" * line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|".format(LABELS[i], *row))
        lines.append("-" * line_len)
    print('\n'.join(lines))


def report_score(actual, predicted):
    score, cm = score_submission(actual, predicted)
    best_score, _ = score_submission(actual, actual)

    print_confusion_matrix(cm)
    print("Score: " + str(score) + " out of " + str(best_score) + "\t(" + str(score * 100 / best_score) + "%)")
    return score * 100 / best_score


def compute_acc(y_true, y_pred):
    if not isinstance(y_true, list) and len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if not isinstance(y_pred, list) and len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    return np.mean(y_true == y_pred)


def write_list(lst, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in lst:
            f.write(str(i) + '\n')
    print('save to:', out_path)


if __name__ == "__main__":
    actual = [0, 0, 0, 0, 1, 1, 0, 1, 1]
    predicted = [0, 0, 0, 0, 1, 1, 0, 1, 0]

    report_score([LABELS[e] for e in actual], [LABELS[e] for e in predicted])
