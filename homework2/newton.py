import argparse
import csv
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.cm as cm
#from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.preprocessing import normalize
import sklearn.metrics
import random


def quad(x):
    f = np.vectorize(lambda elem: elem*elem)
    return f(x)

basis_funcs = {'exp': lambda x: np.exp(quad(x)/(-2)), 'quad': lambda x: quad(x), 'id': lambda x: x}


def sigmoid(a):
    return 1. / (1. + np.exp(-a))

class LinearNewtonClassifier():
    def __init__(self, n_iter=100, reg_coef=None, epsilon=0.001, basis_func='id'):
        self.n_iter = n_iter
        self.w = None
        self.eta = 0.001
        self.reg_coef = reg_coef
        self.basis_func = basis_funcs[basis_func]
        self.epsilon = epsilon
        self.n_features = None

    def __recompute_eta(self, iteration):
        self.eta /= (1. + self.eta * iteration * self.reg_coef)

    def __grad(self, w, xs, ys):
        result = np.zeros(self.n_features)
        for i, x in enumerate(xs):
            result += (sigmoid(np.dot(w, x)) - ys[i]) * self.basis_func(x)
        return result + self.reg_coef * w

    def __hessian(self, w, xs, ys):
        result = self.reg_coef * np.identity(self.n_features)
        for i, x in enumerate(xs):
            a = sigmoid(np.dot(w, x))
            fi = np.atleast_2d(self.basis_func(x))
            m = np.dot(fi, fi.T)
            result += a * m
        return result

    def fit(self, x, y):
        self.n_features = x.shape[1] + 1
        n_samples = x.shape[0]
        self.w = np.zeros(self.n_features)
        self.w[0] = 1.
        if self.reg_coef is None:
            self.reg_coef = 1./n_samples
        x = np.column_stack((np.ones(n_samples), x))

        for k in range(self.n_iter):
            g = self.__grad(self.w, x, y)
            h = self.__hessian(self.w, x, y)
            d = np.linalg.solve(h, -g)
            self.w += self.eta * d
            if (np.fabs(self.eta * d) < self.epsilon).all():
                break
            self.__recompute_eta(k)
        print k
        return self.w

    def predict_proba(self, xs):
        result = []
        if xs.shape[0] == 1:
            xs = np.array(xs)
        xs = np.column_stack((np.ones(xs.shape[0]), xs))
        for x in xs:
            anti_proba = sigmoid(np.dot(self.w, x))
            proba = 1. - anti_proba
            result.append([proba, anti_proba])
        if xs.shape[0] == 1:
            return result[0]
        else:
            return result


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
        title = 'x=attr'+str(inds[0])+' y=attr'+str(inds[1])+' z=attr'+str(inds[2])+' color=attr'+str(inds[3])
        ax.set_title(title)
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
        #ADD NORMALIZATION
        if args.plot_data:
            plot_data_set(data_set, target_set)
            exit(0)

        #solver = 'newton-cg'
        #penalty = 'l2'
        data_set = normalize(data_set)
        clf = {}

        clf[0] = LinearNewtonClassifier(basis_func=args.basis_func)
        clf[1] = LinearNewtonClassifier(basis_func=args.basis_func)
        clf[2] = LinearNewtonClassifier(basis_func=args.basis_func)

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
    parser.add_argument('-f', dest='basis_func', action='store', choices=basis_funcs.keys(), default='id', help='Basis functions')
    parser.add_argument('-k', dest='kfold', action='store', type=int, default=5, help='K-fold for cross-validation')
    return parser.parse_args()


if __name__ == '__main__':
    main()