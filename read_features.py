import numpy as np

# Read Fundamental Frequency
anger_f0 = np.load('fundfreq_anger.npy')
neut_f0 = np.load('fundfreq_neutral.npy')
anger_mfcc = np.load('mfcc_anger.npy')
neut_mfcc = np.load('mfcc_neutral.npy')

print(anger_f0.shape)
print(neut_f0.shape)
print(anger_mfcc.shape)
print(neut_mfcc.shape)

#To access each individual file frequency
print(anger_f0[1].shape)
print(anger_f0[1])
print(anger_f0[2])
print(anger_f0.shape)
print(neut_f0.shape)
print(anger_f0[1]) #To access each individual audiofile fundfreq

# Read MFCCs
anger_mfcc = np.load('mfcc_anger.npy')
neut_mfcc = np.load('mfcc_neutral.npy')
