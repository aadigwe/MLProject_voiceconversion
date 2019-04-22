
#Source: https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b
#https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
import os
import numpy as np
import librosa
import keras
import tensorflow 
import numpy
import scipy
from keras.utils import np_utils
from keras.models import load_model



def wav2mfcc(file_path, max_pad_len=200):
	wave, sr = librosa.load(file_path, mono=True, sr=None)
	wave = wave[::3]
	mfcc = librosa.feature.mfcc(wave, sr=16000)
	pad_width = max_pad_len - mfcc.shape[1]
	mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
	return mfcc

DATA_PATH = "/Users/adaezeadigwe/Desktop/Research/project_ml/Data/"

#A. Get Labels
def get_labels(path=DATA_PATH):
	labels = os.listdir(path)
	label_indices = np.arange(0, len(labels))
	return labels, label_indices, np_utils.to_categorical(label_indices)

labels, label_indices, categories = get_labels(DATA_PATH)
print(labels)
print(label_indices,)
print(categories)

#B. Save MFCCs to .npy files
def save_data_to_array(path=DATA_PATH, max_pad_len=200):
    labels, _, _ = get_labels(path)
    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


#save_data_to_array(path=DATA_PATH, max_pad_len=200)

#C. Split Training and test states  bb
from sklearn.model_selection import train_test_split
def get_train_test(split_ratio=0.6, random_state=42):
    labels, indices, _ = get_labels(DATA_PATH)
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)



#D. Build the RNN model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten

X_train, X_test, y_train, y_test = get_train_test()
#X_train = X_train.reshape(X_train.shape[0], 20*200)
#X_test = X_test.reshape(X_test.shape[0], 20*200)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train.shape[1])

data_dim = 4000
timesteps = 1
num_classes = 5

X_train = np.reshape(X_train, (X_train.shape[0], timesteps, 4000)) #X.reshape(samples, timesteps, features)
X_test = np.reshape(X_test, (X_test.shape[0], timesteps, 4000)) 
y_train_hot = np_utils.to_categorical(y_train)
y_test_hot = np_utils.to_categorical(y_test)


def recurrent_nn():
    model = Sequential()
    model.add(LSTM(units=150,input_shape=(timesteps, data_dim),return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    model.fit(X_train, y_train_hot, batch_size=100, epochs=200, verbose=1, validation_data=(X_test, y_test_hot))
    model.save('rnn_model.h5')
    return model

rnn = recurrent_nn()

score, acc = model.evaluate(X_test, y_test_hot, batch_size=100)
print('Test score:', score)
print('Test accuracy', acc)

model = load_model('rnn_model.h5')


# E. Predict on Reserved Data Set
import csv
def predict_emotion():
    PREDICTION_PATH = "../prediction/"
    with open('predictions_rnn.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Filename','Emotion Prediction','Shape'])
        for sample in os.listdir(PREDICTION_PATH):
            mfcc = wav2mfcc(PREDICTION_PATH + sample)
            shape = mfcc.shape
            # We need to reshape it remember?
            sample_reshaped = mfcc.reshape(1, timesteps, 4000)
            # Perform forward pass
            emotion = get_labels()[0][np.argmax(model.predict(sample_reshaped))]
            emotion = emotion[:-4]
            print(sample + ":" + str(shape) + "," + emotion)
            filewriter.writerow([sample, emotion, shape])

predict_emotion()