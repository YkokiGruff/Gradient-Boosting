#!/usr/bin/env python
# coding: utf-8

# 必要なライブラリをインポートします。詳細はREADME.mdをご覧ださい

# In[]:


import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Kerasに付属の手書き数字画像データをダウンロードします。
# まずnp.random.seed(0)でリセットすると同じ値が生成されるようにします。
# 次にimportした手書き数字データベース60,000枚を訓練用とテスト用それぞれに分けて変数に代入します。
# xが28＊28の画像データで8ビットで保存されています。labaelが0〜9の整数を8ビットで保存しています。

# In[]:


np.random.seed(0)
(X_train_base, labels_train_base), (X_test, labels_test) = mnist.load_data()


# Training set を学習データ（X_train, labels_train）と検証データ（X_validation, labels_validation）に8:2で分割する。（カンマの後にスペースを置くとエラーが発生しますのでご注意ください。）

# In[]:


X_train,X_validation,labels_train,labels_validation = train_test_split(X_train_base,labels_train_base,test_size = 0.2)


# 各画像は行列なので1次元に変換→X_train,X_validation,X_testを上書き
# （28*28＝784の配列になっているのでそれを1行に整列するためそれよりも小さい数字を代入しています。-1784は特に意味はありません。）

# In[]:


X_train = X_train.reshape(-1,784)
X_validation = X_validation.reshape(-1,784)
X_test = X_test.reshape(-1,784)


# 次に正規化です。データタイプを全て不動小数点型に揃えてこれを0.0〜1.0の範囲に入るよう255で割ります。

# In[]:


X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_validation /= 255
X_test /= 255


# 経過時間の計測です。

# In[]:


import time
start = time.time()

