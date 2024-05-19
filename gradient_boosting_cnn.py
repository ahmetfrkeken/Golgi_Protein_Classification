import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

def test(A_train, A_test, B_train, B_test, k):
    a = []
    re = []
    acc = 0
    auc2 = 0
    best_params = {}
    
    for n_estimators in range(35, 61, 2):
        for max_depth in range(1, 11, 2):
            model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(A_train, B_train)
            B_predict = model.predict_proba(A_test)[:, 1]
            B_predict1 = model.predict(A_test)
            
            TP = np.sum((B_predict1 == 1) & (B_test == 1))
            FP = np.sum((B_predict1 == 1) & (B_test == 0))
            TN = np.sum((B_predict1 == 0) & (B_test == 0))
            FN = np.sum((B_predict1 == 0) & (B_test == 1))

            accuracy = accuracy_score(B_test, B_predict1)
            auc1 = roc_auc_score(B_test, B_predict1)
            f1 = f1_score(B_test, B_predict1)
            recall = recall_score(B_test, B_predict1)
            Sn = TP / (TP + FN)
            Sp = TN / (TN + FP)
            
            if acc < accuracy:
                acc = accuracy
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "TP": TP,
                    "FP": FP,
                    "TN": TN,
                    "FN": FN,
                    "accuracy": accuracy,
                    "recall": recall,
                    "auc": auc1,
                    "f1_score": f1,
                    "Sn": Sn,
                    "Sp": Sp
                }

            if auc2 < auc1:
                auc2 = auc1
                best_auc_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "TP": TP,
                    "FP": FP,
                    "TN": TN,
                    "FN": FN,
                    "accuracy": accuracy,
                    "recall": recall,
                    "auc": auc1,
                    "f1_score": f1,
                    "Sn": Sn,
                    "Sp": Sp
                }

            print(f'n_estimators: {n_estimators}, max_depth: {max_depth}, accuracy: {accuracy}, auc: {auc1}, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

    a.append(0)
    np.savetxt(f'acc_record_gradient_boosting_cnn_layer{k}.csv', a, delimiter=',')
    re.append(best_params)
    re.append(best_auc_params)
    np.savetxt(f'evaluate_record_gradient_boosting_cnn_layer{k}.csv', re, delimiter=',', fmt='%s')
    
    print('### Best parameters based on accuracy:', best_params)
    print('### Best parameters based on AUC:', best_auc_params)
    print(f"Model is {type(model)}")

def CNN_layer(train, train_labels, test, test_labels, reshape_height, ii):
    result_cnn = []
    train_num = train.shape[0]
    test_num = test.shape[0]

    imgray = np.expand_dims(train, axis=1)
    imgray = imgray.reshape(train.shape[0], reshape_height, -1)
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
    
    utrain = tf.data.Dataset.zip((input_train, input_train_label)).shuffle(train_num).batch(batch_size=train_num)
    utest = tf.data.Dataset.zip((input_test, input_test_label)).shuffle(test_num).batch(batch_size=test_num)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
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
            monitor='val_accuracy',
            save_best_only=True,
        )
    ]
    
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=METRICS)
    
    history = model.fit(utrain, epochs=1000,
                        validation_data=utest,
                        callbacks=callbacks_list)
    
    df = pd.DataFrame([history.history])
    df.to_csv(f'history_CNN3{ii}.csv')
    best_num = history.history.get('val_accuracy').index(max(history.history.get('val_accuracy')))
    for jj in range(0, 16):
        inde = df.keys()[jj][:]
        result_cnn.extend([df[str(inde)][0][best_num]])

    np.savetxt(f"result_cnn3{ii}.txt", result_cnn)
    print('### max_val_auc = {}'.format(max(history.history.get('val_auc'))))

    model = keras.models.load_model("CNNmodel.h5")
    layer_model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    feature_train = layer_model.predict(utrain)
    feature_test = layer_model.predict(utest)
    
    return feature_train, feature_test

if __name__ == "__main__":
    for i in range(1, 11):
        for k in range(1, 2):
            data = pd.read_csv("UniRep_Train_SMOTE_lgbmSF.csv", encoding="utf-8")
            data = data.iloc[:, 1:]
            testname = pd.read_csv("UniRep_Test_SMOTE_lgbmSF.csv", encoding="utf-8")
            testname = testname.iloc[:, 1:]

            a_train = data.iloc[:, 1:].values
            a_train = preprocessing.Normalizer().fit_transform(a_train)
            a_train = a_train.dot(255)
            b_train = data.iloc[:, 0].values
            a_test = testname.iloc[:, 1:].values
            a_test = preprocessing.Normalizer().fit_transform(a_test)
            a_test = a_test.dot(255)
            b_test = testname.iloc[:, 0].values

            a_train2, a_test2 = CNN_layer(a_train, b_train, a_test, b_test, 10, ii=i)
            
            a_train3 = np.concatenate((a_train, a_train2), axis=1)
            a_test3 = np.concatenate((a_test, a_test2), axis=1)
            a_test2 = preprocessing.Normalizer().fit_transform(a_test2)
            a_train2 = preprocessing.Normalizer().fit_transform(a_train2)
            
            test(a_train2, a_test2, b_train, b_test, i)
