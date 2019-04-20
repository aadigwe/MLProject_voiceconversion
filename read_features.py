import numpy as np

# Read Fundamental Frequency
anger_f0 = np.load('fundfreq_anger.npy')
neut_f0 = np.load('fundfreq_neutral.npy')

print(anger_f0.shape)
print(neut_f0.shape)


#To access each individual file frequency
print(anger_f0[1].shape)