import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
import time
import os
import numpy as np
from datetime import datetime

# SETTINGS
EPOCHS = 15
BATCH_SIZE = 32
NODES = 32
DENSE = 20

dataset_name = 'test1'

def train(dataset_name):

    with open(f'train_data/{dataset_name}-train.pickle', 'rb') as pickle_in:
        train_x, train_y = pickle.load(pickle_in)
    with open(f'train_data/{dataset_name}-val.pickle', 'rb') as pickle_in:
        val_x, val_y = pickle.load(pickle_in)


    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    val_x = np.asarray(val_x)
    val_y = np.asarray(val_y)


    NAME = datetime.now().strftime('%a-%H.%M.%S')

    tensorboard = TensorBoard(log_dir=f"tb_log\\{dataset_name}\\{NAME}")

    model = Sequential()

    model.add(LSTM(NODES, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(NODES, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(NODES, input_shape=(train_x.shape[1:])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(DENSE, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='softmax'))

    model.compile(
            loss='mse',
            optimizer='adam'
                  )

    checkpoint = ModelCheckpoint(filepath=f"models/{dataset_name}/{NAME}/"+ f'e{epoch:02d}-TL{loss:.3f}_VL{val_loss:.3f}.model', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=False, 
            mode='min')
    try:
        os.makedirs(f"models/{dataset_name}/{NAME}")
    except  FileExistsError:
        pass


    history = model.fit(
            train_x, train_y,
            shuffle=True,
            epochs=EPOCHS,
            verbose=2,
            batch_size=BATCH_SIZE,
            validation_data=(validation_x, validation_y),
            callbacks=[tensorboard, checkpoint],
            )



