#Author: Kevin Kipfer

import librosa
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import signal
from scipy.ndimage.morphology import binary_erosion, binary_dilation
#import matplotlib.pyplot as plt

#plt.style.use('ggplot')

maxValue = 1.7
minValue = -1.8

#########################################################################################
#
# Data Augmentation
# z.B. http://ofai.at/~jan.schlueter/code/augment/
#
#########################################################################################


def pitch_shift(signal, sr, n_steps=5):
  '''
  
  '''

  signal = librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)

  return signal


def time_stretch(signal, rate):
  '''
  Input:
    signal; sound signal to be stretched
    rate; stretch factor: if rate < 1 then signal is slowed down, otherwise sped up
  Output:
    stretched/compressed signal
  CAUTION: changes time length of signal -> apply this before extract_signal_of_importance, consider cough window size
  '''
  
  signal = librosa.effects.time_stretch(y=signal, rate=rate)
  
  return signal


def time_shift(spect):
  '''
  Input:
    Spectrogram to be augmented
  Output:
    Spectrogram cut into two pieces along time dimension. Then second part is placed before the first
  '''
  spect_length = spect.shape[1]
  idx = np.random.randint(int(spect_length*0.1), int(spect_length*0.9))
  spect_ = np.hstack([spect[:,idx:], spect[:,:idx]])

  return spect_


def add_noise(signal):
  '''
  Input:
    sound signal; time series vector, standardized
  Output:
    sound signal + gaussian noise
  '''
  std = 0.05 * np.max(signal)
  noise_mat = np.random.randn(signal.shape[0])*std
  return signal + noise_mat


def denoise_spectrogram(spect, threshold=1, filter_size = (2,2)):
  """
  input:
    spectrogram, matrix
  output:
    denoised spectrogram, binary matrix as in bird singing paper
  """

  # map to [0,1]
  minVal = np.min(spect)
  maxVal = np.max(spect)
  spect = (spect - minVal)/(maxVal - minVal)

  # convert to binary
  row_medians = np.tile(np.median(spect, axis=1, keepdims=True), (1, spect.shape[1]))
  col_medians = np.tile(np.median(spect, axis=0, keepdims=True), (spect.shape[0], 1))
  spect_ = (spect > threshold * row_medians).astype('int') * (spect > threshold * col_medians).astype('int')
  
  # apply erosion + dilation
  structure_filter = np.ones(filter_size)
  spect_ = binary_erosion(spect_, structure=structure_filter)
  spect_ = binary_dilation(spect_, structure=structure_filter)

  return spect_



#########################################################################################
#
# Extracting Features
#
#########################################################################################

def standardize(timeSignal):

	 #TODO
         maxValue_ = np.max(timeSignal)
         minValue_ = np.min(timeSignal)
         timeSignal = (timeSignal - minValue)/(maxValue - minValue) 

         #but since timeSignal is in [-1.8,1.7]
         #timeSignal /= 1.8
         return timeSignal


def extract_Signal_Of_Importance(signal, window, sample_rate ):
        """
	extract a window around the maximum of the signal
	input: 	signal
                window -> size of a window
		sample_rate 
        """

        window_size = int(window * sample_rate)			

        start = max(0, np.argmax(np.abs(signal)) - (window_size // 2))
        end = min(np.size(signal), start + window_size)
        signal = signal[start:end]

        length = np.size(signal)
        assert length <= window_size, 'extracted signal is longer than the allowed window size'
        if length < window_size:
                #pad zeros to the signal if too short
                signal = np.concatenate((signal, np.zeros(window_size-length))) 
        return signal


def fetch_samples(files, 
		  is_training=True, 
                  hop_length=224,#112,#120#56
		  bands = 16,
		  window = 0.16,
                  do_denoise=False):
	"""
	load, preprocess, normalize a sample
	input: a list of strings
	output: the processed features from each sample path in the input
	"""
	batch_features = []
	for f in files:
                try:
                       timeSignal, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')
                except ValueError as e:
                       print ('!!!!!!! librosa failed to load file: %s !!!!!!!!!'%f)
                       raise e

                timeSignal = extract_Signal_Of_Importance(timeSignal, window, sample_rate)

                #fit_scale(timeSignal)
                timeSignal = standardize(timeSignal)

                mfcc = librosa.feature.melspectrogram(y=timeSignal, sr=sample_rate, n_mels=bands, power=1, hop_length=hop_length)

                if do_denoise:
                  mfcc = denoise_spectrogram(mfcc)

                batch_features.append(mfcc)

	return np.array(batch_features)



















