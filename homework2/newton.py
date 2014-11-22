import argparse
import csv
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import sklearn.metrics
import random

def plot_data_set(x, y):
    print "Plotting data"
    fig = pl.figure(figsize=(30, 15))
    x1 = np.transpose(x[:50])
    x2 = np.transpose(x[50:100])
    x3 = np.transpose(x[100:150])

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        colors = x[:, i]
        norm = np.linalg.norm(colors)
        cmin = colors.min()
        cmax = colors.max()

        inds = np.array([0, 1, 2, 3])
        inds = np.delete(inds, [i])
        inds = np.append(inds, i)

        print inds
        ax.scatter(x1[inds[0]], x1[inds[1]], x1[inds[2]], c=x1[inds[3]], marker='o', vmin=cmin, vmax=cmax, label='class1', cmap=cm.brg)
        ax.scatter(x2[inds[0]], x2[inds[1]], x2[inds[2]], c=x2[inds[3]], marker='D', vmin=cmin, vmax=cmax, label='class2', cmap=cm.brg)
        ax.scatter(x3[inds[0]], x3[inds[1]], x3[inds[2]], c=x3[inds[3]], marker='+', vmin=cmin, vmax=cmax, label='class3', cmap=cm.brg)
    pl.show()


def vote_predict(predict):
    result = []
    for i in range(len(predict[0])):
        max_prob = predict[0][i][0]
        max_class = 0
        if max_prob < predict[1][i][0]:
            max_prob = predict[1][i][0]
            max_class = 1
        if max_prob < predict[2][i][0]:
            #max_prob = predict[2][i][0]
            max_class = 2
        result.append(max_class)
    return result


def main():
    with open('iris.data', 'rb') as file:
        ls = list(csv.reader(file))
        args = parse_args()
        k = args.kfold
        data_set = ls

        def translate_name():
            return { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2 }
        target_set = [translate_name()[x[4]] for x in data_set]
        data_set = map(lambda xl: [float(a) for a in xl[:-1]], data_set)
        data_set = np.array(data_set)
        target_set = np.array(target_set)

        if args.plot_data:
            plot_data_set(data_set, target_set)
            exit(0)

        solver = 'newton-cg'
        penalty = 'l2'
        clf = {}
        clf[0] = LogisticRegression(C=data_set.shape[0], solver=solver, penalty=penalty)
        clf[1] = LogisticRegression(C=data_set.shape[0], solver=solver, penalty=penalty)
        clf[2] = LogisticRegression(C=data_set.shape[0], solver=solver, penalty=penalty)
        kfolds = KFold(data_set.shape[0], n_folds=k, shuffle=True, random_state=random.randint(1, 251192))
        predict = {}
        precisions = []
        recalls = []
        accuracies = []
        for i, (train_indices, test_indices) in enumerate(kfolds):
            x_test = data_set[test_indices]
            y_test = target_set[test_indices]
            for i in range(3):
                x_train = data_set[train_indices]
                y_train = target_set[train_indices]

                for ti, t in enumerate(train_indices):
                    if y_train[ti] == i:
                        y_train[ti] = 0
                    else:
                        y_train[ti] = 1
                clf[i].fit(x_train, y_train)
                predict[i] = clf[i].predict_proba(x_test)

            y_predict = vote_predict(predict)
            precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(y_test, y_predict)
            accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
            precisions.append(precision)
            recalls.append(recall)
            accuracies.append(accuracy)

        precision = np.array(precisions).mean()
        recall = np.array(recalls).mean()
        accuracy = np.array(accuracies).mean()

        print 'precision =', precision
        print 'recall =', recall
        print 'accuracy =', accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Experiments on linear classifiers applied to Iris data set')

    parser.add_argument('-p', dest='plot_data', action='store_true', default=False, help='Plot data')
    parser.add_argument('-f', dest='basis func', action='store', choices=['exp', 'quad', 'id'], default='id', help='Basis functions')
    parser.add_argument('-k', dest='kfold', action='store', type=int, default=5, help='K-fold for cross-validation')
    return parser.parse_args()


if __name__ == '__main__':
    main()