import lightgbm as lgb
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# 以下Catboostのコードを参照してください
np.random.seed(0)
(X_train_base, labels_train_base), (X_test, labels_test) = mnist.load_data()

X_train,X_validation,labels_train,labels_validation = train_test_split(X_train_base,labels_train_base,test_size = 0.2)

X_train = X_train.reshape(-1,784)
X_validation = X_validation.reshape(-1,784)
X_test = X_test.reshape(-1,784)

X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_validation /= 255
X_test /= 255

train_data = lgb.Dataset(X_train, label=labels_train)
eval_data = lgb.Dataset(X_validation, label=labels_validation, reference= train_data)

import time
start = time.time()


#light gbm モデル構築
params = {
'task': 'train',
'boosting_type': 'gbdt',
'objective': 'multiclass',
'num_class': 10,
}

gbm = lgb.train(
params,
train_data,
valid_sets=eval_data,
num_boost_round=100,
early_stopping_rounds=10,
)

preds = gbm.predict(X_test)
preds
y_pred = []
for x in preds:
  y_pred.append(np.argmax(x))

from sklearn.metrics import accuracy_score
print('accuracy_score:{}'.format(accuracy_score(labels_test, y_pred)))

#経過時間
print('elapsed_timetime:{}'.format(time.time()-start))
