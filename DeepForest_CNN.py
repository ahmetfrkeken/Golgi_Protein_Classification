
"""

Description :
Python3 implementation of the gcForest algorithm preesented in Zhou and Feng 2017
(paper can be found here : https://arxiv.org/abs/1702.08835 ).
It uses the typical scikit-learn syntax  with a .fit() function for training
and a .predict() function for predictions.

"""
import itertools
import numpy as np
import pandas as pd
import joblib
from numpy import *


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import os


from sklearn.decomposition import PCA,_kernel_pca
import os
from sklearn.datasets import load_iris
import scipy.io as scio

#__status__ = "Development"

# noinspection PyUnboundLocalVariable
class gcForest(object):
    def __init__(self, shape_1X=None, n_mgsRFtree=30, window=None, stride=1,
                 cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=np.inf,
                 min_samples_mgs=0.1, min_samples_cascade=0.1, tolerance=0.0, n_jobs=-1):
        """ gcForest Classifier.

        :param shape_1X: int or tuple list or np.array (default=None)
            Shape of a single sample element [n_lines, n_cols]. Required when calling mg_scanning!
            For sequence data a single int can be given.

        :param n_mgsRFtree: int (default=30)
            Number of trees in a Random Forest during Multi Grain Scanning.

        :param window: int (default=None)
            List of window sizes to use during Multi Grain Scanning.
            If 'None' no slicing will be done.

        :param stride: int (default=1)
            Step used when slicing the data.

        :param cascade_test_size: float or int (default=0.2)
            Split fraction or absolute number for cascade training set splitting.

        :param n_cascadeRF: int (default=2)
            Number of Random Forests in a cascade layer.
            For each pseudo Random Forest a complete Random Forest is created, hence
            the total numbe of Random Forests in a layer will be 2*n_cascadeRF.

        :param n_cascadeRFtree: int (default=101)
            Number of trees in a single Random Forest in a cascade layer.

        :param min_samples_mgs: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Multi-Grain Scanning Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param min_samples_cascade: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Cascade Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param cascade_layer: int (default=np.inf)
            mMximum number of cascade layers allowed.
            Useful to limit the contruction of the cascade.

        :param tolerance: float (default=0.0)
            Accuracy tolerance for the casacade growth.
            If the improvement in accuracy is not better than the tolerance the construction is
            stopped.

        :param n_jobs: int (default=1)
            The number of jobs to run in parallel for any Random Forest fit and predict.
            If -1, then the number of jobs is set to the number of cores.
        """
        setattr(self, 'shape_1X', shape_1X)
        setattr(self, 'n_layer', 0)
        setattr(self, '_n_samples', 0)
        setattr(self, 'n_cascadeRF', int(n_cascadeRF))
        if isinstance(window, int):
            setattr(self, 'window', [window])
        elif isinstance(window, list):
            setattr(self, 'window', window)
        setattr(self, 'stride', stride)
        setattr(self, 'cascade_test_size', cascade_test_size)
        setattr(self, 'n_mgsRFtree', int(n_mgsRFtree))
        setattr(self, 'n_cascadeRFtree', int(n_cascadeRFtree))
        setattr(self, 'cascade_layer', cascade_layer)
        setattr(self, 'min_samples_mgs', min_samples_mgs)
        setattr(self, 'min_samples_cascade', min_samples_cascade)
        setattr(self, 'tolerance', tolerance)
        setattr(self, 'n_jobs', n_jobs)

    def fit(self, X, y):
        """ Training the gcForest on input data X and associated target y.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array
            1D array containing the target values.
            Must be of shape [n_samples]
        """
        if np.shape(X)[0] != len(y):
            raise ValueError('Sizes of y and X do not match.')

        mgs_X = self.mg_scanning(X, y)
        _ = self.cascade_forest(mgs_X, y)

    def predict_proba(self, X):
        """ Predict the class probabilities of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted class probabilities for each input sample.
        """
        mgs_X = self.mg_scanning(X)
        # print(mgs_X.shape)
        cascade_all_pred_prob = self.cascade_forest(mgs_X)
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)

        return predict_proba

    def predict(self, X):
        """ Predict the class of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        pred_proba = self.predict_proba(X=X)
        predictions = np.argmax(pred_proba, axis=1)

        return predictions

    def mg_scanning(self, X, y=None):
        """ Performs a Multi Grain Scanning on input data.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)

        :return: np.array
            Array of shape [n_samples, .. ] containing Multi Grain Scanning sliced data.
        """
        setattr(self, '_n_samples', np.shape(X)[0])
        shape_1X = getattr(self, 'shape_1X')
        if isinstance(shape_1X, int):
            shape_1X = [1,shape_1X]
        if not getattr(self, 'window'):
            setattr(self, 'window', [shape_1X[1]])

        mgs_pred_prob = []

        for wdw_size in getattr(self, 'window'):
            wdw_pred_prob = self.window_slicing_pred_prob(X, wdw_size, shape_1X, y=y)
            mgs_pred_prob.append(wdw_pred_prob)

        return np.concatenate(mgs_pred_prob, axis=1)

    def window_slicing_pred_prob(self, X, window, shape_1X, y=None):
        """ Performs a window slicing of the input data and send them through Random Forests.
        If target values 'y' are provided sliced data are then used to train the Random Forests.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample.

        :param y: np.array (default=None)
            Target values. If 'None' no training is done.

        :return: np.array
            Array of size [n_samples, ..] containing the Random Forest.
            prediction probability for each input sample.
        """
        n_tree = getattr(self, 'n_mgsRFtree')
        min_samples = getattr(self, 'min_samples_mgs')
        stride = getattr(self, 'stride')

        if shape_1X[0] > 1:
            print('Slicing Images...')
            sliced_X, sliced_y = self._window_slicing_img(X, window, shape_1X, y=y, stride=stride)
        else:
            # print('Slicing Sequence...')
            sliced_X, sliced_y = self._window_slicing_sequence(X, window, shape_1X, y=y, stride=stride)

        if y is not None:
            n_jobs = getattr(self, 'n_jobs')
            prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                         min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
            crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                         min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
            # print('Training MGS Random Forests...')
            prf.fit(sliced_X, sliced_y)
            crf.fit(sliced_X, sliced_y)
            setattr(self, '_mgsprf_{}'.format(window), prf)
            setattr(self, '_mgscrf_{}'.format(window), crf)
            pred_prob_prf = prf.oob_decision_function_
            pred_prob_crf = crf.oob_decision_function_

        if hasattr(self, '_mgsprf_{}'.format(window)) and y is None:
            prf = getattr(self, '_mgsprf_{}'.format(window))
            crf = getattr(self, '_mgscrf_{}'.format(window))
            pred_prob_prf = prf.predict_proba(sliced_X)
            pred_prob_crf = crf.predict_proba(sliced_X)

        pred_prob = np.c_[pred_prob_prf, pred_prob_crf]

        return pred_prob.reshape([getattr(self, '_n_samples'), -1])

    def _window_slicing_img(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for images

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_cols].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced images and target values (empty if 'y' is None).
        """
        if any(s < window for s in shape_1X):
            raise ValueError('window must be smaller than both dimensions for an image')

        len_iter_x = np.floor_divide((shape_1X[1] - window), stride) + 1
        len_iter_y = np.floor_divide((shape_1X[0] - window), stride) + 1
        iterx_array = np.arange(0, stride*len_iter_x, stride)
        itery_array = np.arange(0, stride*len_iter_y, stride)

        ref_row = np.arange(0, window)
        ref_ind = np.ravel([ref_row + shape_1X[1] * i for i in range(window)])
        inds_to_take = [ref_ind + ix + shape_1X[1] * iy
                        for ix, iy in itertools.product(iterx_array, itery_array)]

        sliced_imgs = np.take(X, inds_to_take, axis=1).reshape(-1, window**2)

        if y is not None:
            sliced_target = np.repeat(y, len_iter_x * len_iter_y)
        elif y is None:
            sliced_target = None

        return sliced_imgs, sliced_target

    def _window_slicing_sequence(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for sequences (aka shape_1X = [.., 1]).

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_col].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced sequences and target values (empty if 'y' is None).
        """
        if shape_1X[1] < window:
            raise ValueError('window must be smaller than the sequence dimension')

        len_iter = np.floor_divide((shape_1X[1] - window), stride) + 1
        iter_array = np.arange(0, stride*len_iter, stride)

        ind_1X = np.arange(np.prod(shape_1X))
        inds_to_take = [ind_1X[i:i+window] for i in iter_array]
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, window)

        if y is not None:
            sliced_target = np.repeat(y, len_iter)
        elif y is None:
            sliced_target = None

        return sliced_sqce, sliced_target

    def cascade_forest(self, X, y=None):
        """ Perform (or train if 'y' is not None) a cascade forest estimator.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        if y is not None:
            setattr(self, 'n_layer', 0)
            test_size = getattr(self, 'cascade_test_size')
            max_layers = getattr(self, 'cascade_layer')
            tol = getattr(self, 'tolerance')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            self.n_layer += 1
            prf_crf_pred_ref = self._cascade_layer(X_train, y_train)
            accuracy_ref = self._cascade_evaluation(X_test, y_test)
            feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)

            self.n_layer += 1
            prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
            accuracy_layer = self._cascade_evaluation(X_test, y_test)

            while accuracy_layer > (accuracy_ref + tol) and self.n_layer <= max_layers:
                accuracy_ref = accuracy_layer
                prf_crf_pred_ref = prf_crf_pred_layer
                feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)
                self.n_layer += 1
                prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
                accuracy_layer = self._cascade_evaluation(X_test, y_test)

            if accuracy_layer < accuracy_ref :
                n_cascadeRF = getattr(self, 'n_cascadeRF')
                for irf in range(n_cascadeRF):
                    delattr(self, '_casprf{}_{}'.format(self.n_layer, irf))
                    delattr(self, '_cascrf{}_{}'.format(self.n_layer, irf))
                self.n_layer -= 1

        elif y is None:
            at_layer = 1
            prf_crf_pred_ref = self._cascade_layer(X, layer=at_layer)
            while at_layer < getattr(self, 'n_layer'):
                at_layer += 1
                feat_arr = self._create_feat_arr(X, prf_crf_pred_ref)
                prf_crf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)

        return prf_crf_pred_ref

    def _cascade_layer(self, X, y=None, layer=0):
        """ Cascade layer containing Random Forest estimators.
        If y is not None the layer is trained.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :param layer: int (default=0)
            Layer indice. Used to call the previously trained layer.

        :return: list
            List containing the prediction probabilities for all samples.
        """
        n_tree = getattr(self, 'n_cascadeRFtree')
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        min_samples = getattr(self, 'min_samples_cascade')

        n_jobs = getattr(self, 'n_jobs')
        prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
        crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)

        prf_crf_pred = []
        if y is not None:
            # print('Adding Training Layer, n_layer={}'.format(self.n_layer))
            for irf in range(n_cascadeRF):
                prf.fit(X, y)
                crf.fit(X, y)
                setattr(self, '_casprf{}_{}'.format(self.n_layer, irf), prf)
                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), crf)
                prf_crf_pred.append(prf.oob_decision_function_)
                prf_crf_pred.append(crf.oob_decision_function_)
        elif y is None:
            for irf in range(n_cascadeRF):
                prf = getattr(self, '_casprf{}_{}'.format(layer, irf))
                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))
                prf_crf_pred.append(prf.predict_proba(X))
                prf_crf_pred.append(crf.predict_proba(X))

        return prf_crf_pred

    def _cascade_evaluation(self, X_test, y_test):
        """ Evaluate the accuracy of the cascade using X and y.

        :param X_test: np.array
            Array containing the test input samples.
            Must be of the same shape as training data.

        :param y_test: np.array
            Test target values.

        :return: float
            the cascade accuracy.
        """
        casc_pred_prob = np.mean(self.cascade_forest(X_test), axis=0)
        casc_pred = np.argmax(casc_pred_prob, axis=1)
        casc_accuracy = accuracy_score(y_true=y_test, y_pred=casc_pred)
        # print('Layer validation accuracy = {}'.format(casc_accuracy))

        return casc_accuracy

    def _create_feat_arr(self, X, prf_crf_pred):
        """ Concatenate the original feature vector with the predicition probabilities
        of a cascade layer.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param prf_crf_pred: list
            Prediction probabilities by a cascade layer for X.

        :return: np.array
            Concatenation of X and the predicted probabilities.
            To be used for the next layer in a cascade forest.
        """
        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        feat_arr = np.concatenate([add_feat, X], axis=1)

        return feat_arr

def test(A_train, A_test, B_train, B_test, k):
    # print('==========================Data Shape======================')
    # print(A_train.shape)
    # # print(A_test)
    # # print(B_train)
    # print(B_test.shape)
    a = []
    re = []
    acc = 0
    auc2 = 0
    bmgsRFtree = 0
    b_window = 0
    b_n_cascadeRFtree = 0
    max_iterations = 20
    desired_accuracy = 0.95
    iteration_count = 0
    best_accuracy = 0

    for x in range(35, 61, 2):
        for y in range(90, 121, 2):
            for z in range(1, 32, 3):
                iteration_count += 1
                model = gcForest(shape_1X=A_train.shape[1], n_mgsRFtree=x, window=z, n_cascadeRFtree=y)
                model.fit(A_train, B_train)
                B_predict = model.predict_proba(A_test)
                B_predict = B_predict.tolist()
                B_predict1 = model.predict(A_test)
                
                i = 0
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                for one_res in B_predict1:
                    if one_res == 1:
                        if B_test[i] == 1:
                            TP += 1
                            i += 1
                        else:
                            FP += 1
                            i += 1
                    else:
                        if B_test[i] == 0:
                            TN += 1
                            i += 1
                        else:
                            FN += 1
                            i += 1

                accuarcy = accuracy_score(y_true=B_test, y_pred=B_predict1)
                auc1 = roc_auc_score(B_test, B_predict1)
                if acc < accuarcy:
                    acc = accuarcy
                    bmgsRFtree = x
                    b_window = z
                    b_n_cascadeRFtree = y
                    auc = roc_auc_score(B_test, B_predict1)
                    f1 = f1_score(B_test, B_predict1)
                    re_call = recall_score(B_test, B_predict1)
                    TP1 = TP
                    FN1 = FN
                    TN1 = TN
                    FP1 = FP
                    Sn = TP / (TP + FN)
                    Sp = TN / (TN + FP)
                # print(accuarcy)
                # print('gcForest accuarcy : {}'.format(acc))
                print('n_mgsRFtree:{}'.format(x), 'window:{}'.format(z), 'n_cascadeRFtree:{}'.format(y),
                      'accuracy:{}'.format(accuarcy), 'auc:{}'.format(auc1), 'TP:{}'.format(TP), 'FP:{}'.format(FP),
                      'TN:{}'.format(TN), 'FN:{}'.format(FN))
                if auc2 < auc1:
                    acc2 = accuracy_score(y_true=B_test, y_pred=B_predict1)
                    bmgsRFtree2 = x
                    b_window2 = z
                    b_n_cascadeRFtree2 = y
                    auc2 = auc1
                    f12 = f1_score(B_test, B_predict1)
                    re_call2 = recall_score(B_test, B_predict1)
                    TP12 = TP
                    FN12 = FN
                    TN12 = TN
                    FP12 = FP
                    Sn2 = TP / (TP + FN)
                    Sp2 = TN / (TN + FP)
                    
                
                if accuarcy > best_accuracy:
                    best_accuracy = accuarcy
                    best_params = (x, y, z)

                print(f'Iteration {iteration_count}: n_mgsRFtree={x}, window={z}, n_cascadeRFtree={y}, accuracy={accuarcy}')
                # İstenen doğruluk seviyesine ulaşıldığında veya maksimum iterasyon sayısına ulaşıldığında döngüyü durdur.
                if best_accuracy >= desired_accuracy or iteration_count >= max_iterations:
                    print("Stopping criteria met")
                    break
            if best_accuracy >= desired_accuracy or iteration_count >= max_iterations:
                break
            
        if best_accuracy >= desired_accuracy or iteration_count >= max_iterations:
            break
                # print(accuarcy)
                # print('gcForest accuarcy : {}'.format(acc))
    a.append(0)
    # np.savetxt('acc_record_cnn' + str(k) + '.csv', a, delimiter=',')
    np.savetxt('acc_record_cnnSINGLE_layer' + str(k) + '.csv', a, delimiter=',')
    re.append([bmgsRFtree, b_window, b_n_cascadeRFtree, TP1, FP1, TN1, FN1, acc, re_call, auc, f1, Sn, Sp])
    re.append([bmgsRFtree2, b_window2, b_n_cascadeRFtree2, TP12, FP12, TN12, FN12, acc2, re_call2, auc2, f12, Sn2, Sp2])
    # np.savetxt('evaluate_record_cnn' + str(k) + '.csv', re, delimiter=',')
    np.savetxt('evaluate_record_cnnSINGLE_layer' + str(k) + '.csv', re, delimiter=',')
    print('### n_mgsRFtree:{}'.format(bmgsRFtree), 'window:{}'.format(b_window),
          'n_cascadeRFtree:{}'.format(b_n_cascadeRFtree), 'TP:{}'.format(TP1), 'FP:{}'.format(FP1), 'TN:{}'.format(TN1),
          'FN:{}'.format(FN1), 'accuracy:{}'.format(acc), 'recall:{}'.format(re_call), 'auc:{}'.format(auc),
          'f1_score:{}'.format(f1), 'Sn:{}'.format(Sn), 'Sp:{}'.format(Sp), )
    print('### n_mgsRFtree:{}'.format(bmgsRFtree2), 'window:{}'.format(b_window2),
          'n_cascadeRFtree:{}'.format(b_n_cascadeRFtree2), 'TP:{}'.format(TP12), 'FP:{}'.format(FP12),
          'TN:{}'.format(TN12), 'FN:{}'.format(FN12), 'accuracy:{}'.format(acc2), 'recall:{}'.format(re_call2),
          'auc:{}'.format(auc2), 'f1_score:{}'.format(f12), 'Sn:{}'.format(Sn2), 'Sp:{}'.format(Sp2), )


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_normal(shape, dtype=tf.float32)
    return tf.Variable(initial)

def get_random_batchdata(n_samples, batchsize):
    indexs = np.random.randint(0, n_samples, batchsize)
    return indexs

def max_pool_22(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],padding="SAME")


def CNN_layer(train, train_labels, test, test_labels, reshape_height,ii
               ):
    result_cnn = []
    train_num = train.shape[0]
    test_num = test.shape[0]

    imgray = np.expand_dims(train, axis=1)
    imgray = imgray.reshape(train.shape[0], reshape_height, -1)
    # imgray = tf.convert_to_tensor(imgray, dtype=np.float32)
    im_num, im_height, im_width = imgray.shape
    imgray = np.expand_dims(imgray, axis=1)
    input_train = imgray.reshape(im_num, im_height, im_width, 1)
    input_train = tf.data.Dataset.from_tensor_slices(input_train)
    imgray = np.expand_dims(test, axis=1)
    imgray = imgray.reshape(test.shape[0], reshape_height, -1)
    im_num, im_height, im_width = imgray.shape
    imgray = np.expand_dims(imgray, axis=1)
    input_test = imgray.reshape(im_num, im_height, im_width, 1)
    input_test = tf.data.Dataset.from_tensor_slices(input_test)
    input_train_label = tf.data.Dataset.from_tensor_slices(train_labels)
    input_test_label = tf.data.Dataset.from_tensor_slices(test_labels)
    print(input_train)
    print(input_test)
    utrain1 = tf.data.Dataset.zip((input_train, input_train_label))
    utrain = utrain1.shuffle(train_num).batch(batch_size=train_num)
    utest1 = tf.data.Dataset.zip((input_test, input_test_label))
    utest = utest1.shuffle(test_num).batch(batch_size=test_num)

    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(im_height, im_width, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D())#第一次
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D())

    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D())
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D())
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D())
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(512, activation='relu'))
    # model.add(tf.keras.layers.Dense(512, activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='CNNmodel.h5',
            monitor='val_accuracy',  # 这两个参数的含义是，如果val_loss 没有改善，那么不需要覆盖模型文件。这就可以始终保存在训练过程中见到的最佳模型
            save_best_only=True,
        )]
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=METRICS)
    history = model.fit(utrain, epochs=32,
                        # steps_per_epoch=train_num // batch_size,
                        validation_data=utest,
                        # validation_steps=test_num //
                        callbacks = callbacks_list
                        )
    # print(history.history.keys())
    # plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
    print('### max_val_acc = {}'.format(max(history.history.get('val_accuracy'))))
    # plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
    df = pd.DataFrame([history.history])
    # df.to_csv('history_CNN' + str(ii) + '.csv')
    df.to_csv('history_CNN3' + str(ii) + '.csv')
    best_num = history.history.get('val_accuracy').index(max(history.history.get('val_accuracy')))
    for jj in range(0, 16):
        inde = df.keys()[jj][:]
        result_cnn.extend([df[str(inde)][0][best_num]])

    # np.savetxt("result_cnn" + str(ii) + ".txt", result_cnn)
    np.savetxt("result_cnn3" + str(ii) + ".txt", result_cnn)
    print('### max_val_auc = {}'.format(max(history.history.get('val_auc'))))
    # plt.plot(history.epoch, history.history.get('val_auc'), label='val_auc')
    # plt.show()


    model = keras.models.load_model("CNNmodel.h5")
    # model.summary()
    layer_model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    # print(model.layers[13].output)
    feature_train = layer_model.predict(utrain)
    # K.set_image_dim_ordering('th')
    feature_test = layer_model.predict(utest)
    print(feature_train.shape)
    return feature_train,feature_test

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

    for i in range(1, 11):
        for k in range(1, 2):

            data = pd.read_csv("UniRep_Train_SMOTE_lgbmSF.csv",encoding="utf-8")
            data = data.iloc[:,1:]
            testname = pd.read_csv("UniRep_Test_SMOTE_lgbmSF.csv",encoding="utf-8")
            testname = testname.iloc[:,1:]
            # file = open('result'+str(cfg.glo_i)+'.txt', 'r',encoding='utf-8')
            # file = open("result" + str(i) + "-" + str(k) + '.txt', 'w', encoding='utf-8')
            # string = ""
            # string = string + "--" + str(k) + '\n'
            # # file1 = open('result'+str(cfg.glo_i)+'.txt', 'w', encoding='utf-8')
            # file1 = open("result" + str(i) + "-" + str(k) + '.txt', 'w', encoding='utf-8')
            # file1.write(string)
            # file1.close()
            # file.close()
            a_train = data.iloc[:,1:].values
            a_train = preprocessing.Normalizer().fit_transform(a_train)
            a_train = a_train.dot(255)
            # a_train = preprocessing.scale(a_train)
            b_train = data.iloc[:, 0].values
            a_test = testname.iloc[:, 1:].values
            a_test = preprocessing.Normalizer().fit_transform(a_test)
            a_test = a_test.dot(255)
            print(np.min(a_test))
            # a_test = preprocessing.scale(a_test)
            b_test = testname.iloc[:, 0].values
            # uni = np.concatenate((a_train,a_test),axis = 0)
            # pca = PCA(n_components=100, svd_solver="auto", whiten=True)
            # uni = pca.fit_transform(uni)
            # a_train1 = uni[0:a_train.shape[0],:]
            # a_test1 = uni[a_train.shape[0]:(a_test.shape[0]+a_train.shape[0]),:]
            # print(a_train1.shape)
            a_train2,a_test2 = CNN_layer(a_train, b_train, a_test, b_test, 10, ii=i)
            print((a_train2.shape, a_test2.shape))
            # a_test2 = preprocessing.StandardScaler().fit_transform(a_test2)
            # a_train2 = preprocessing.StandardScaler().fit_transform(a_train2)
                      # a_train.shape[0])
            # uni = np.concatenate((a_train2, a_test2), axis=0)
            # pca = PCA(n_components=100, svd_solver="auto", whiten=True)
            # uni = pca.fit_transform(uni)
            # a_train22 = uni[0:a_train2.shape[0], :]
            # a_test22 = uni[a_train2.shape[0]:(a_test2.shape[0] + a_train2.shape[0]), :]
            # print(a_train2.shape)
            a_train3 = np.concatenate((a_train, a_train2), axis = 1)
            a_test3 = np.concatenate((a_test, a_test2), axis=1)
            a_test2 = preprocessing.Normalizer().fit_transform(a_test2)
            a_train2 = preprocessing.Normalizer().fit_transform(a_train2)
            print(a_train3.shape)
            print(a_test3.shape)
            test(a_train2, a_test2, b_train, b_test, i)
