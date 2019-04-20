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

def save_fundfreq_to_array(path = Data_Directory):
	labels, _, _ = get_labels(path)
	print(labels)
	for label in labels:
		fundfreq_vectors = []
		wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
		for wavfile in wavfiles:
			print(wavfile)
			x, fs = sf.read(wavfile)
			_f0, t = pw.dio(x, fs)
			x, fs = sf.read(path)
			f0 = pw.stonemask(x, _f0, t, fs)
			fundfreq_vectors.append(f0)
		np.save(label + '.npy', fundfreq_vectors)

save_fundfreq_to_array(Data_Directory)

#B. LOOP to extract vector by sample of target fundamental frequency

#C. Perform Linear Regression

##############################################################

#A. LOOP to extract vector by sample of source MFCC

#B. LOOP to extract vector by sample of target MFCC

#C. Perform Linear Regression

##############################################################

#A. LOOP to extract vector by sample of source Spectral envelope

#B. LOOP to extract vector by sample of target Spectral Envelope

#C. Perform Linear Regression


