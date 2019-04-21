import numpy as np

# Read Fundamental Frequency
anger_f0 = np.load('fundfreq_anger.npy')
neut_f0 = np.load('fundfreq_neutral.npy')
print(anger_f0.shape)
print(neut_f0.shape)
print(anger_f0[1]) #To access each individual audiofile fundfreq

# Read MFCCs
anger_mfcc = np.load('mfcc_anger.npy')
neut_mfcc = np.load('mfcc_neutral.npy')
