__author__ = 'rim'

import numpy
from sklearn.datasets import load_iris

class CartTree():
    def __init__(self, min_leaf_size):
        self.tree = {}
        if min_leaf_size <= 0:
            min_leaf_size = 1
        self.min_leaf_size = min_leaf_size

    def fit(self, data_set, target_set):
        self.tree = self.__fit_recursive(data_set, target_set)

    def __fit_recursive(self, data_set, target_set):
        n_features = data_set.shape[1]
        n_samples = data_set.shape[0]

        tree = {'target': target_set.mean(), 'leaf': True}

        parent_impurity = self.__loss_function(target_set)

        best_delta_impurity, best_right_set, best_left_set = None, None, None
        best_left_target_set, best_right_target_set = None, None
        best_feature, best_value = None, None

        for feature in range(n_features):
            average_values = numpy.array(list(self.__compute_average_values(data_set[:, feature])))

            for value in average_values:
                left_indices = [i for i, x in enumerate(target_set) if data_set[i, feature] < value]
                right_indices = [i for i, x in enumerate(target_set) if data_set[i, feature] > value]

                left_set = data_set[left_indices]
                left_target_set = target_set[left_indices]
                right_set = data_set[right_indices]
                right_target_set = target_set[right_indices]

                current_delta_impurity = self.__delta_impurity(parent_impurity, self.__loss_function(left_target_set),
                                                             self.__loss_function(right_target_set), data_set.shape[0],
                                                             left_set.shape[0], right_set.shape[0])
                if best_delta_impurity < current_delta_impurity:
                    best_delta_impurity = current_delta_impurity
                    best_right_set, best_left_set = right_set, left_set
                    best_left_target_set, best_right_target_set = left_target_set, right_target_set
                    best_feature, best_value = feature, value

        if best_left_set is None or best_right_set is None \
                or min(best_left_set.shape[0], best_right_set.shape[0]) < self.min_leaf_size:
            return tree

        tree['leaf'] = False
        tree['predicate'] = (best_feature, best_value)
        tree['left'] = self.__fit_recursive(best_left_set, best_left_target_set)
        tree['right'] = self.__fit_recursive(best_right_set, best_right_target_set)
        return tree

    def __loss_function(self, target_set):
        return numpy.sum(numpy.power(target_set - target_set.mean(), 2))

    def __delta_impurity(self, parent_impurity, left_impurity, right_impurity, n, nl, nr):
        return parent_impurity - (float(nl) / n * left_impurity) - (float(nr) / n * right_impurity)

    def __compute_average_values(self, feature_column):
        ui_vector = numpy.unique(numpy.sort(feature_column))
        ui_iter = iter(ui_vector)
        ui0 = ui_iter.next()
        for i in range(ui_vector.shape[0]-1):
            ui1 = ui_iter.next()
            yield (ui0 + ui1) / float(2)
            ui0 = ui1

    def predict(self, xs):
        return numpy.array([self.__predict_recursive(self.tree, x) for x in xs])

    def __predict_recursive(self, tree, x):
        if tree['leaf']:
            return tree['target']
        else:
            feature, value = tree['predicate']
            if x[feature] <= value:
                return self.__predict_recursive(tree['left'], x)
            else:
                return self.__predict_recursive(tree['right'], x)

def main():
    data_set = load_iris()['data']
    target_set = load_iris()['target']

    cartTree = CartTree(min_leaf_size=5)

    cartTree.fit(data_set, target_set)

    print cartTree.tree
    print target_set
    print numpy.array([int(round(cartTree.predict([x]))) for x in data_set])

if __name__ == '__main__':
    main()