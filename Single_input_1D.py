import keras
from keras import Input, layers, optimizers, regularizers, backend
from keras.models import Model, load_model
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
import data_partition
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import KFold
from keras.optimizers import rmsprop_v2

# 一维time序列模型
input_time = Input(shape=(256 * 5, 23))
first_block_conv = layers.Conv1D(filters=16, kernel_size=3, padding='same')(input_time)
first_block_relu = layers.LeakyReLU(alpha=0.1)(first_block_conv)
first_block_batchNormalization = layers.BatchNormalization()(first_block_relu)
first_block_conv = layers.Conv1D(filters=16, kernel_size=3, padding='same')(input_time)
first_block_relu = layers.LeakyReLU(alpha=0.1)(first_block_conv)
first_block_batchNormalization = layers.BatchNormalization()(first_block_relu)
first_block_maxpooling = layers.MaxPooling1D(pool_size=2)(first_block_batchNormalization)

second_block_conv = layers.Conv1D(filters=32, kernel_size=3, padding='same')(first_block_maxpooling)
second_block_relu = layers.LeakyReLU(alpha=0.1)(second_block_conv)
second_block_batchNormalization = layers.BatchNormalization()(second_block_relu)
second_block_conv = layers.Conv1D(filters=32, kernel_size=3, padding='same')(first_block_maxpooling)
second_block_relu = layers.LeakyReLU(alpha=0.1)(second_block_conv)
second_block_batchNormalization = layers.BatchNormalization()(second_block_relu)
second_block_maxpooling = layers.MaxPooling1D(pool_size=2)(second_block_batchNormalization)

third_block_conv = layers.Conv1D(filters=64, kernel_size=3, padding='same')(second_block_maxpooling)
third_block_relu = layers.LeakyReLU(alpha=0.1)(third_block_conv)
third_block_batchNormalization = layers.BatchNormalization()(third_block_relu)
third_block_conv = layers.Conv1D(filters=64, kernel_size=3, padding='same')(second_block_maxpooling)
third_block_relu = layers.LeakyReLU(alpha=0.1)(third_block_conv)
third_block_batchNormalization = layers.BatchNormalization()(third_block_relu)
third_block_maxpooling = layers.MaxPooling1D(pool_size=2)(third_block_batchNormalization)

# 连接两个输入
# concatenated = layers.concatenate([third_block_maxpooling, third_block_maxpooling_tf], axis=1)  # axis=1是指对行进行拼接

bilstm_first_tf = layers.Bidirectional(layers.LSTM(20, return_sequences=True))(third_block_maxpooling)
bilstm_second_tf = layers.Bidirectional(layers.LSTM(20, return_sequences=True))(bilstm_first_tf)

flatten = layers.Flatten()(bilstm_first_tf)
dropout = layers.Dropout(0.3)(flatten)
dense = layers.Dense(128, activation='relu')(dropout)
# dense = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(dropout)
# dropout = layers.Dropout(0.3)(dense)
output = layers.Dense(1, activation='sigmoid')(dense)

# 数据准备
patients = ["12"]
firstPartPath = 'F:\CHB-MIT\chb-mit-scalp-eeg-database-1.0.0\\'
simplePath = 'D:\sun\paperProject\dataset'
a = np.load(simplePath + '\patient' + patients[0] + '\\negativeSimP5.npy')
b = np.load(simplePath + '\patient' + patients[0] + '\positiveSimP5.npy')

positiveLabel = np.ones(np.size(b, 0), dtype=int)
negativeLabel = np.zeros(np.size(a, 0), dtype=int)

# 合并阳性和阴性的数据集
simple = np.vstack((a, b))
label = np.hstack((negativeLabel, positiveLabel))

# 数据滤波处理
simpleNum = np.size(simple, 0)
secondDim = np.size(simple, 1)
b, a = signal.butter(4, [2 * 0.01 / 256, 2 * 32 / 256], 'bandpass')  # 4为滤波器的阶数，
simple = signal.filtfilt(b, a, simple, axis=2)

# 按照行进行标准化
for i in range(simpleNum):
    for j in range(secondDim):
        simple[i, j, :] -= simple[i, j, :].mean()
        simple[i, j, :] /= simple[i, j, :].std()

half_simple = math.floor(0.5 * np.size(simple, 0))  # 正样本或者负样本的数量
split = 5
kf = KFold(n_splits=split)
k = 0
test_loss = np.zeros(split)
test_acc = np.zeros(split)
y_pred_base = []
y_true_base = []
y_pred = []
y_true = []

for train_index, test_index in kf.split(simple[:half_simple]):
    # 训练集
    x_train_negative = simple[train_index]
    x_train_positive = simple[half_simple + train_index]
    x_train = np.vstack((x_train_negative, x_train_positive))

    # 训练集标签
    y_train_negative = label[train_index]
    y_train_positive = label[half_simple + train_index]
    y_train = np.hstack((y_train_negative, y_train_positive))

    # 测试集
    x_test_negative = simple[test_index]
    x_test_positive = simple[half_simple + test_index]
    x_test = np.vstack((x_test_negative, x_test_positive))

    # 测试集标签
    y_test_negative = label[test_index]
    y_test_positive = label[half_simple + test_index]
    y_test = np.hstack((y_test_negative, y_test_positive))

    # 从训练集中划分出验证集
    trainNum = math.floor(0.5 * 0.8 * np.size(x_train, 0))  # 训练集和验证集的比例为 90：10
    # trainNum = math.floor(0.5 * 0.1 * np.size(simple, 0))  # 验证集正样本或者负样本的数量为  10%
    partial_x_train_posi = x_train[:trainNum]
    partial_x_train_nega = x_train[
                           math.floor(0.5 * np.size(x_train, 0)):math.floor(0.5 * np.size(x_train, 0)) + trainNum]
    partial_x_train = np.vstack((partial_x_train_posi, partial_x_train_nega))

    x_val_posi = x_train[trainNum:math.floor(0.5 * np.size(x_train, 0))]
    x_val_nega = x_train[math.floor(0.5 * np.size(x_train, 0)) + trainNum:]
    x_val = np.vstack((x_val_posi, x_val_nega))

    partial_y_train_posi = y_train[:trainNum]
    partial_y_train_nega = y_train[
                           math.floor(0.5 * np.size(x_train, 0)):math.floor(0.5 * np.size(x_train, 0)) + trainNum]
    partial_y_train = np.hstack((partial_y_train_posi, partial_y_train_nega))

    y_val_posi = y_train[trainNum:math.floor(0.5 * np.size(x_train, 0))]
    y_val_nega = y_train[math.floor(0.5 * np.size(x_train, 0)) + trainNum:]
    y_val = np.hstack((y_val_posi, y_val_nega))

    # 扰乱训练集数据
    data_num = len(partial_x_train)
    index = np.arange(data_num)
    np.random.shuffle(index)
    partial_x_train = partial_x_train[index]
    partial_y_train = partial_y_train[index]

    # 扰乱测试集数据
    data_num = len(x_test)
    index = np.arange(data_num)
    np.random.shuffle(index)
    x_test = x_test[index]
    y_test = y_test[index]

    # 扰乱验证集数据
    data_num = len(x_val)
    index = np.arange(data_num)
    np.random.shuffle(index)
    x_val = x_val[index]
    y_val = y_val[index]

    # 准备时间序列数据
    partial_x_train_t = partial_x_train.transpose((0, 2, 1))
    x_test_t = x_test.transpose((0, 2, 1))
    x_val_t = x_val.transpose((0, 2, 1))

    # 准备时频数据
    fs = 256  # 采样频率

    # 对训练数据进行 STFT
    f, t, nd_train = signal.stft(partial_x_train, fs=256, window='hann', nperseg=128)
    p_train = np.abs(nd_train)
    partial_x_train = p_train
    partial_x_train_tf = partial_x_train.transpose((0, 2, 3, 1))

    # 对验证数据进行 STFT
    f, t, nd_val = signal.stft(x_val, fs=256, window='hann', nperseg=128)
    p_val = np.abs(nd_val)
    x_val = p_val
    x_val_tf = x_val.transpose((0, 2, 3, 1))

    # 对测试数据进行 STFT
    f, t, nd_test = signal.stft(x_test, fs=256, window='hann', nperseg=128)
    p_test = np.abs(nd_test)
    x_test = nd_test
    x_test_tf = x_test.transpose((0, 2, 3, 1))

    # 实例化模型
    model = Model(input_time, output)
    # model.summary()

    # 使用回调函数，早停
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='acc',
            patience=10,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='patient01_11.h5',
            monitor='val_loss',
            save_best_only=True,
        )
    ]

    # 编译模型
    model.compile(loss='binary_crossentropy',
                  optimizer=rmsprop_v2.RMSProp(learning_rate=1e-4),
                  metrics=['acc'])
    # 模型参数的微调：
    # 优化器：
    # 损失函数：
    # 训练模型
    print('第', k + 1, '折：')
    history = model.fit(partial_x_train_t,
                        partial_y_train,
                        epochs=100,
                        batch_size=32,
                        callbacks=callbacks_list,
                        validation_data=(x_val_t, y_val))

    test_loss[k], test_acc[k] = model.evaluate(x_test_t, y_test)
    result_base = model.predict(x_test_t)
    y_pred_base.append(result_base)
    y_true_base.append(y_test)
    # k = k + 1

    # 保存模型
    model.save(simplePath + '\patient' + patients[0] + '_2.h5')

    # # 加载模型
    conv_base = load_model(simplePath + '\patient' + patients[0] + '_2.h5')
    conv_base.trainable = True
    #
    # # 微调模型
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'bidirectional':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # # # 连接训练集和验证集
    x_train_t = np.vstack((partial_x_train_t, x_val_t))
    x_train_tf = np.vstack((partial_x_train_tf, x_val_tf))

    y_train = np.hstack((partial_y_train, y_val))

    y_true.append(y_test)
    history = conv_base.fit(x_train_t,
                            y_train,
                            epochs=30,
                            batch_size=32)
    # 测试模型
    test_loss[k], test_acc[k] = conv_base.evaluate(x_test_t, y_test)
    result = conv_base.predict(x_test_t)
    y_pred.append(result)
    k = k + 1

print(test_loss.mean())
print(test_acc.mean())

# 基网络
np.save(simplePath + '\patient' + patients[0] + '\\onesecond\\base\ypred.npy', y_pred_base)
np.save(simplePath + '\patient' + patients[0] + '\\onesecond\\base\ytrue.npy', y_true_base)

# 微调模型
np.save(simplePath + '\patient' + patients[0] + '\\onesecond\\finetuning\ypred.npy', y_pred)
np.save(simplePath + '\patient' + patients[0] + '\\onesecond\\finetuning\ytrue.npy', y_true)
a = 0
