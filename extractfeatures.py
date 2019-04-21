import os
import argparse
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import soundfile as sf
import pyworld as pw
import librosa
import scipy.io.wavfile as wav 
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from keras.utils import np_utils



##############################################################
# Test one audio file
path='/Users/adaezeadigwe/Desktop/Research/project_ml/Data/anger/anger_0001.wav'
Data_Directory = '/Users/adaezeadigwe/Desktop/Research/project_ml/Data/'

x, fs = sf.read(path)
_f0, t = pw.dio(x, fs)    # raw pith extractor
f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
mfcc = librosa.feature.mfcc(x, sr=16000)
y = pw.synthesize(f0, sp, ap, fs)
#wav.write('neutral_syn.wav',fs, y)
print(f0.shape)	  #1071,
print(sp.shape)	  #1071 , 513	
print(ap.shape)	  #1071, 513
print(mfcc.shape) #20,168

##############################################################

#A. LOOP to extract vector by sample of source fundamental frequency

def get_labels(path=Data_Directory):
	labels = os.listdir(path)
	label_indices = np.arange(0, len(labels))
	return labels, label_indices, np_utils.to_categorical(label_indices)

def wav2mfcc(file_path, max_pad_len=120):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def save_features_to_array(path = Data_Directory):
	labels, _, _ = get_labels(path)
	print(labels)
	for label in labels:
		fundfreq_vectors = []
		ap_vectors = []
		mfcc_vectors = []
		sp_vectors = []

		wavfiles = [path + label + '/' + wavfile for wavfile in sorted(os.listdir(path + '/' + label))]
		for wavfile in wavfiles:
			print(wavfile)
			x, fs = sf.read(wavfile)

			_f0, t = pw.dio(x, fs)
			f0 = pw.stonemask(x, _f0, t, fs)
			fundfreq_vectors.append(f0)

			sp = pw.cheaptrick(x, f0, t, fs)
			sp_vectors.append(sp)

			ap = pw.d4c(x, f0, t, fs)
			ap_vectors.append(ap)

			mfcc = wav2mfcc(wavfile, max_pad_len=120)
			mfcc_vectors.append(mfcc)
		#print(mfcc_vectors.shape)
		np.save('mfcc_' + label + '.npy', mfcc_vectors)
		np.save('fundfreq_' + label + '.npy', fundfreq_vectors)
		#np.save('sp_' + label + '.npy', sp_vectors)
		#np.save('ap_' + label + '.npy', ap_vectors)
		#np.save('mfcc_' + label + '.npy', mfcc_vectors)

save_features_to_array(Data_Directory)


#C. Split Training and test states
def get_train_test(split_ratio=0.6, random_state=42):
	labels, indices, _ = get_labels(DATA_PATH)
	X = np.load(labels[0] + '.npy')
	y = np.zeros(X.shape[0])


#B. LOOP to extract vector by sample of target fundamental frequency

#C. Perform Linear Regression

##############################################################

#A. LOOP to extract vector by sample of target MFCC

#B. Perform Linear Regression

##############################################################

#A. LOOP to extract vector by sample of source Spectral envelope

#B. LOOP to extract vector by sample of target Spectral Envelope

#C. Perform Linear Regression


