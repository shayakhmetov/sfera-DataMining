__author__ = 'rim'

from scipy.optimize import minimize_scalar
import numpy
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from CART import CartTree
import random
from sklearn.metrics import accuracy_score


def log_e():
    return numpy.log2(numpy.exp(1))


class StochasticGradientBoosting():
    def __init__(self, n_trees=2, min_leaf_size=5, rsm=0.2, bagging=1, regularization_coef=0.1):
        self.n_trees = n_trees
        self.min_leaf_size = min_leaf_size
        self.composition = []
        self.f0 = 0.
        self.num_labels = 0
        self.rsm = rsm
        self.bagging = bagging
        self.regularization_coef = regularization_coef

    def fit(self, data_set, target_set):
        labels = numpy.unique(target_set)
        self.f0 = minimize_scalar(lambda gamma: self.__logistic_loss_function(target_set, gamma)).x
        other_data_set, other_target_set = data_set, target_set
        current_data_set, current_target_set = data_set, target_set
        for m in range(self.n_trees):
            if other_target_set.shape[0] < 2:
                return None
            if self.bagging < 1.:
                other_data_set, current_data_set, other_target_set, current_target_set =\
                    train_test_split(data_set, target_set, test_size=self.bagging)
            rm = - self.__gradient_loss_function(current_target_set, self.__decision_func(current_data_set))
            cart_tree = CartTree(self.min_leaf_size)
            cart_tree.fit(current_data_set, rm)
            gamma_m = minimize_scalar(lambda gamma:
                                      self.__logistic_loss_function(current_target_set, self.__sigmoid(
                                          self.__decision_func(current_data_set) +
                                          gamma*cart_tree.predict(current_data_set))), method='bounded', bounds=(-100, 100))
            self.composition.append((self.regularization_coef*gamma_m.x, cart_tree))

    def __logistic_loss_function(self, y, fx):
        sig = self.__sigmoid(fx)
        #return - numpy.sum(y*numpy.log2(sig) + (1 - y)*numpy.log2(1 - sig))
        return numpy.sum(-y*fx*log_e() - numpy.log2(1. - sig))

    def __sigmoid(self, x):
        sig = 1. / (1. + numpy.exp(-x))
        return sig

    def __gradient_loss_function(self, y, fx):
        return numpy.log2(numpy.exp(-y)) + self.__sigmoid(-fx)


    def __decision_func(self, xs):
        decision = numpy.full(xs.shape[0], self.f0)
        for b, tree in self.composition:
            decision += b*tree.predict(xs)
        return decision

    def predict(self, xs):
        return self.__sigmoid(self.__decision_func(xs))

    def print_b(self):
        print '{',
        for b, tree in self.composition:
            print b,
        print '}'


def convert_targets(label, targets):
    for l in targets:
        if l == label:
            yield 1
        else:
            yield 0


def construct_predicted(pd1, pd2, pd3):
    for i in range(pd1.shape[0]):
        l = numpy.argmax([pd1[i], pd2[i], pd3[i]])
        yield l

def main():
    data_set = load_iris()['data']
    target_set = load_iris()['target']
    min_n_trees = 1
    max_n_trees = 50
    min_leaf_size = 5
    max_leaf_size = 6

    train_data_set, test_data_set, train_target_set, test_target_set = \
                train_test_split(data_set, target_set, test_size=0.2)

    for leaf_size in range(min_leaf_size, max_leaf_size, 1):
        for m in range(min_n_trees, max_n_trees, 5):
            classifier1 = StochasticGradientBoosting(n_trees=m+1, min_leaf_size=leaf_size)
            classifier2 = StochasticGradientBoosting(n_trees=m+1, min_leaf_size=leaf_size)
            classifier3 = StochasticGradientBoosting(n_trees=m+1, min_leaf_size=leaf_size)

            targets1 = numpy.array(list(convert_targets(0, train_target_set)))
            targets2 = numpy.array(list(convert_targets(1, train_target_set)))
            targets3 = numpy.array(list(convert_targets(2, train_target_set)))

            classifier1.fit(train_data_set, targets1)
            classifier2.fit(train_data_set, targets2)
            classifier3.fit(train_data_set, targets3)

            print '-----!!! ', m+1, ' Trees:'
            classifier1.print_b()
            classifier2.print_b()
            classifier3.print_b()
            predicted1 = classifier1.predict(test_data_set)
            predicted2 = classifier2.predict(test_data_set)
            predicted3 = classifier3.predict(test_data_set)
            #print predicted1
            #print predicted2
            #print predicted3
            print 'ideal = ', numpy.array(test_target_set)
            predicted = numpy.array(list(construct_predicted(predicted1, predicted2, predicted3)))
            print 'predict=', predicted
            print 'RESULT =', accuracy_score(test_target_set, predicted), ' with min_leaf_size =', leaf_size
            print

if __name__ == '__main__':
    main()