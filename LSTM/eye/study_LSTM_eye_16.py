#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy
import sys
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import glob
import optuna
import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from matplotlib.colors import ListedColormap

from keras import regularizers
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, LSTM, Flatten, Dense, Conv1D, MaxPooling1D, AveragePooling1D, Input, concatenate, Add, LayerNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras_self_attention import SeqSelfAttention

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from tqdm.notebook import tqdm


def load_data(length, filename):

    # データの読み込み
    args = sys.argv
    path = "/home/sho/catkin_ws/src/ros_whill/script/"  #データの保存場所
    all_files = glob.glob(path + filename + ".xlsx")
    #all_files = glob.glob(path + "/study_all.xlsx")
    li = []
    for filename in all_files:
        frame = pd.read_excel(filename)
        li.append(frame)
    df = pd.concat(li, axis=0, ignore_index=True)

    # 値の変更
    df.loc[(-90 <= df['JoystickAngle']) & (df['JoystickAngle'] < -15), 'JoystickAngle2'] = -1
    df.loc[(-15 <= df['JoystickAngle']) & (df['JoystickAngle'] <= 15), 'JoystickAngle2'] = 0
    df.loc[(15 < df['JoystickAngle']) & (df['JoystickAngle'] <= 90), 'JoystickAngle2'] = 1
    df.loc[(90 < df['JoystickAngle']), 'JoystickAngle2'] = 2
    df.loc[(10 <= df['FaceHistgramTime']), 'FaceHistgramTime'] = 10
    df.loc[(10 <= df['EyeHistgramTime']), 'EyeHistgramTime'] = 10
    df['JoystickAngle2']+=1
     
    L = len(df)

    # 入力データ、出力データ作成
    JoystickAngle     = np.array([df['JoystickAngle2']]) # 列を抽出し、numpy配列に変換する。
    face              = np.array([df['Face']])
    faceEuclid        = np.array([df['FaceEuclid']])
    faceHistgramTime  = np.array([df['FaceHistgramTime']])
    faceStandardDev   = np.array([df['FaceStandardDev']])
    faceScan          = np.array([df['scanFace']])
    eye               = np.array([df['Eye']])
    eyeEuclid         = np.array([df['EyeEuclid']])
    eyeHistgramTime   = np.array([df['EyeHistgramTime']])
    eyeStandardDev    = np.array([df['EyeStandardDev']])
    eyeScan           = np.array([df['scanGaze']])

    #入力データの正規化
    face              = zscore(face, 3243600, 0, 1, 0) 
    faceEuclid        = zscore(faceEuclid, 70, 0, 1, 0)
    faceHistgramTime  = zscore(faceHistgramTime, 10, 0, 1, 0)
    faceStandardDev   = zscore(faceStandardDev, 25, 0, 1, 0)
    faceScan          = zscore(faceScan, 6, 0, 1, 0)
    eye               = zscore(eye, 3243600, 0, 1, 0)
    eyeEuclid         = zscore(eyeEuclid, 90, 0, 1, 0)
    eyeHistgramTime   = zscore(eyeHistgramTime, 10, 0, 1, 0)
    eyeStandardDev    = zscore(eyeStandardDev, 25, 0, 1, 0)
    eyeScan           = zscore(eyeScan, 6, 0, 1, 0)

    #行列に変換する。（配列の要素数行×1列）
    y = JoystickAngle.reshape(-1, 1)  
    face             = face.reshape(-1, 1)
    faceEuclid       = faceEuclid.reshape(-1, 1)
    faceHistgramTime = faceHistgramTime.reshape(-1, 1)
    faceStandardDev  = faceStandardDev.reshape(-1, 1)
    faceScan         = faceScan.reshape(-1, 1)
    eye              = eye.reshape(-1, 1)
    eyeEuclid        = eyeEuclid.reshape(-1, 1)
    eyeHistgramTime  = eyeHistgramTime.reshape(-1, 1)
    eyeStandardDev   = eyeStandardDev.reshape(-1, 1)
    eyeScan          = eyeScan.reshape(-1, 1)

    #時系列分のゼロ行列を用意
    a = np.zeros((L-length+1, 1))  
    b = np.zeros((L-length+1, 1))
    c = np.zeros((L-length+1, 1))
    d = np.zeros((L-length+1, 1))
    e = np.zeros((L-length+1, 1))
    f = np.zeros((L-length+1, 1))
    g = np.zeros((L-length+1, 1))
    h = np.zeros((L-length+1, 1))
    j = np.zeros((L-length+1, 1))
    k = np.zeros((L-length+1, 1))
    
    #ゼロ行列に時系列データを代入
    for i in range(length):        
     a = np.concatenate([a, face[i:L-(length-1-i), :]], axis=1)
     b = np.concatenate([b, faceEuclid[i:L-(length-1-i), :]], axis=1)
     c = np.concatenate([c, faceHistgramTime[i:L-(length-1-i), :]], axis=1)
     d = np.concatenate([d, faceStandardDev[i:L-(length-1-i), :]], axis=1)
     e = np.concatenate([e, faceScan[i:L-(length-1-i), :]], axis=1)
     f = np.concatenate([f, eye[i:L-(length-1-i), :]], axis=1)
     g = np.concatenate([g, eyeEuclid[i:L-(length-1-i), :]], axis=1)
     h = np.concatenate([h, eyeHistgramTime[i:L-(length-1-i), :]], axis=1)
     j = np.concatenate([j, eyeStandardDev[i:L-(length-1-i), :]], axis=1)
     k = np.concatenate([k, eyeScan[i:L-(length-1-i), :]], axis=1)
     
    #ゼロ行列削除
    a = np.delete(a, 0, axis=1) 
    b = np.delete(b, 0, axis=1)
    c = np.delete(c, 0, axis=1)
    d = np.delete(d, 0, axis=1)
    e = np.delete(e, 0, axis=1)
    f = np.delete(f, 0, axis=1)
    g = np.delete(g, 0, axis=1)
    h = np.delete(h, 0, axis=1)
    j = np.delete(j, 0, axis=1)
    k = np.delete(k, 0, axis=1)
    
    #numpy配列を結合する。
    x = np.stack([f, g, h, j, k], axis=1)
    y = y[length-1:L, :]  # 予測対象日の当日のデータ

    #reshape
    x = np.reshape(x, [x.shape[0], x.shape[2], x.shape[1]])
    y = np.reshape(y, [y.shape[0],])
    print(x.shape)
    print(y.shape)

    #学習用とテスト用にデータを分割
    #x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
     
    return x, y
    



def zscore(x, Max, Min, M, m):

    y = (x-Min)/(Max-Min)
    zscore = y * (M-m) + m
    
    return zscore



def main():

    #部分時系列のリスト
    length = 16

    index = 0         #モデル重み保存用の識別子
    
    #データの読み込み
    x, y = load_data(length, "study_all13")
    x_valid, y_valid = load_data(length, "test_all6")
    
    # accuracy, f_score保存の配列
    test_accuracy_val = []  #生データ
    test_accuracy_mean = [] #平均
    test_accuracy_std = []  #標準偏差
    test_f_val = []
    test_f_mean = []
    test_f_std = []
    
    valid_accuracy_val = []  #生データ
    valid_accuracy_mean = [] #平均
    valid_accuracy_std = []  #標準偏差
    valid_f_val = []
    valid_f_mean = []
    valid_f_std = []
    
    #乱数シードを固定値で初期化
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    #入力、出力のサイズ
    inputs = Input(shape=(x.shape[1], x.shape[2]))
    output_dim = 4                #出力データの次元数：同上

    #ミニバッチサイズ
    BATCHSIZE = 50
    
    # 学習エポック数
    EPOCHS = 100
    
    #ハイパーパラメータ with maxpooling
    lstm_layer     = 1           #LSTM層の数
    dense_layer    = 2           #全結合層の数
    lstm_units     = [48]    #LSTM層1のユニット数
    dense_nodes    = [32, 32]        #全結合層1のユニット数
    dropout_rate   = 0.0        #ドロップアウト率
    learning_rate  = 0.001       #学習率
    
    #時系列分割交差検証
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=5, test_size=5000)
    for train_index, test_index in tscv.split(x, y):
     x_train, x_test = x[train_index], x[test_index]
     y_train, y_test = y[train_index], y[test_index]
    
     #モデル構築
     #LSTM
     p = LSTM(lstm_units[0], dropout=dropout_rate, return_sequences=True)(inputs)
     for i in range(1, lstm_layer):
      p = LSTM(lstm_units[i], dropout=dropout_rate, return_sequences=True)(p)
      #p = LayerNormalization()(p)
    
     p = Flatten()(p)
    
     #Fully_connect
     for i in range(dense_layer):
       p = Dense(units=dense_nodes[i], activation="relu")(p)
       p = LayerNormalization()(p)
       p = Dropout(dropout_rate)(p)
      
     #softmax
     p = Dense(units=output_dim, activation="softmax")(p)
    
     #Model compile
     model = Model(inputs=inputs, outputs=p)
     model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
     model.summary()

     #過学習の抑制
     early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1)
     
     #val_accuracyの改善が3エポック見られなかったら、学習率を0.5倍する。
     reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=0.0001)

     #評価に用いるモデルの重みデータの保存
     model_weights = "LSTM_eye_" + str(length) + "_" + str(index) + ".h5"
     check_point = ModelCheckpoint(model_weights, monitor='val_accuracy', verbose=1, save_best_only=True)
     
     
     #学習開始
     history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCHSIZE, epochs=EPOCHS, callbacks=[early_stopping, reduce_lr, check_point], verbose=1)

     # Pandas 形式
     result = pd.DataFrame(history.history)

     # 目的関数の値
     #result[['loss', 'val_loss']].plot();

     # 正解率
     #result[['accuracy', 'val_accuracy']].plot();
     #plt.show()

     #モデルのロード
     model = load_model(model_weights)
     #h5形式からpb形式に変換して保存
     directory = "saved_model" + str(length) + "/1dcnnlstm_model" + str(index)
     model.save(directory)
     model.summary()
     
     #テストデータで予測
     test_predicted  = model.predict(x_test)
     valid_predicted = model.predict(x_valid)
     
     # 空のリストを定義
     test_array_predicted = []
     valid_array_predicted = []
     valid_n = []

     # 最も値の大きなラベルを取得
     for i in range(len(test_predicted)):
      test_p = np.argmax(test_predicted[i])
      test_array_predicted.append(test_p)
      
     for i in range(len(valid_predicted)):
      valid_n.append(valid_predicted[i])
      valid_p = np.argmax(valid_predicted[i])
      valid_array_predicted.append(valid_p)
      
     #グラフ
     #plt.figure(figsize=(20,15))
     #plt.plot(y_valid, label = 'result')
     #plt.plot(valid_array_predicted, label = 'predicted')
     #plt.legend(loc='best')
     #plt.show()

     #混合行列
     #cmatrix = confusion_matrix(y_valid, valid_array_predicted)
     #sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')
     #sns.heatmap(cmatrix, square = True, cbar = True, annot = True, cmap = 'Greens')
     #plt.xlabel('predictedLabel')
     #plt.ylabel('trueLabel');
     #plt.show()
      
     #評価値
     test_acc_score = accuracy_score(y_test, test_array_predicted)
     test_f_score = f1_score(y_test, test_array_predicted, average="macro")
     test_accuracy_val.append(test_acc_score*100)
     test_f_val.append(test_f_score)

     valid_acc_score = accuracy_score(y_valid, valid_array_predicted)
     valid_f_score = f1_score(y_valid, valid_array_predicted, average="macro")
     valid_accuracy_val.append(valid_acc_score*100)
     valid_f_val.append(valid_f_score)
    
     print(classification_report(y_test, test_array_predicted))
     print(classification_report(y_valid, valid_array_predicted))
     
     print("test_accuracy_val: {}" .format(test_accuracy_val))
     print("vali_accuracy_val: {}" .format(valid_accuracy_val))
     print("test_f_val: {}" .format(test_f_val))
     print("vali_f_val: {}" .format(valid_f_val))
     print()
     
     
     ### COMPUTE FEATURE IMPORTANCE ###
     results = []
     COLS = ['Eye', 'EyeEuclid', 'EyeHistgramTime', 'EyeStandardDev', 'scanGaze']
     print(' Computing LSTM feature importance...')

     # COMPUTE BASELINE (NO SHUFFLE)
     scce = tf.keras.losses.SparseCategoricalCrossentropy()
     baseline_ce = scce(y_valid, valid_n).numpy()
     results.append({'feature':'BASELINE','ce':baseline_ce})           

     # COMPUTE OOF CE WITH FEATURE K SHUFFLED
     for k in range(len(COLS)):
    
         # 空のリストを定義
         valid_n = []
    
         # SHUFFLE FEATURE K
         save_col = x_valid[:,:,k].copy()
         np.random.shuffle(x_valid[:,:,k])
                            
         # COMPUTE OOF CE WITH FEATURE K SHUFFLED
         valid_predicted = model.predict(x_valid)
         for i in range(len(valid_predicted)):
             valid_n.append(valid_predicted[i])
            
         scce = tf.keras.losses.SparseCategoricalCrossentropy()
         ce = scce(y_valid, valid_n).numpy()
         results.append({'feature':COLS[k],'ce':ce})
         x_valid[:,:,k] = save_col
             
     # DISPLAY LSTM FEATURE IMPORTANCE
     print()
     df = pd.DataFrame(results)
     df = df.sort_values('ce')
     plt.figure(figsize=(10,20))
     plt.barh(np.arange(len(COLS)+1),df.ce)
     plt.yticks(np.arange(len(COLS)+1),df.feature.values)
     plt.title('LSTM Feature Importance',size=16)
     plt.ylim((-1,len(COLS)+1))
     plt.plot([baseline_ce,baseline_ce],[-1,len(COLS)+1], '--', color='orange',label=f'Baseline OOF\nCE={baseline_ce:.3f}')
     plt.xlabel(f'Fold {index+1} OOF CE with feature permuted',size=14)
     plt.ylabel('Feature',size=14)
     plt.legend()
     #plt.show()
                                   
     # SAVE LSTM FEATURE IMPORTANCE
     df = df.sort_values('ce',ascending=False)
     df.to_csv(f'lstm_feature_importance_fold_{index+1}.csv',index=False)

     index += 1
    
    test_accuracy_mean.append(np.mean(test_accuracy_val))
    test_accuracy_std.append(np.std(test_accuracy_val))
    test_f_mean.append(np.mean(test_f_val))
    test_f_std.append(np.std(test_f_val))
         
    valid_accuracy_mean.append(np.mean(valid_accuracy_val))
    valid_accuracy_std.append(np.std(valid_accuracy_val))
    valid_f_mean.append(np.mean(valid_f_val))
    valid_f_std.append(np.std(valid_f_val))
        
    print("test_accuracy_mean: {}" .format(test_accuracy_mean))
    print("vali_accuracy_mean: {}" .format(valid_accuracy_mean))
    print("test_accuracy_std: {}" .format(test_accuracy_std))
    print("vali_accuracy_std: {}" .format(valid_accuracy_std))
    print("test_f_mean: {}" .format(test_f_mean))
    print("vali_f_mean: {}" .format(valid_f_mean))
    print("test_f_std: {}" .format(test_f_std))
    print("vali_f_std: {}" .format(valid_f_std))



if __name__ == '__main__':
    main()
