#Copyright (C) 2020 Bastien Lechat

#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
#published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
#of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

import pywt
import numpy as np
from KC_algorithm.utils import nextpow2, pad_nextpow2

class wave:
    """
    Class to deal with discrete wavelet transform of an epoch
    """
    def __init__(self,Fs,wav,epochs):
        self.fs = Fs
        self.wavelet = pywt.Wavelet(wav)
        self.Epochs = epochs
        self.ntrials, self.len = np.shape(epochs)
        #self.len = np.shape(epochs)[1]
        self.keptscales = 0
        self.CoefMatrix = 0
        self.denoised_coefs = 0
        self.slices = list()


    def get_waveletinfo(self):
        print(self.wavelet)

    def showdwtfreqs(self):
        scales = MaxScale(self.len, self.wavelet)
        level = scales+1

        for k in np.arange(scales):
            level =  level - 1
            down_freq = self.fs / (2 ** (k + 1))
            up_freq = self.fs / (2 ** k)
            print('Details coefficient {}: {} to {} Hz'.format(level, up_freq, down_freq))
        print('Approximation coefficient {}: {} to {} Hz'.format(level, up_freq / 2, 0))

    def dwtdec(self):

        scales = MaxScale(self.len, self.wavelet)
        samples = get_samples(self.len, scales)
        CoefMatrix = np.zeros((self.ntrials,np.sum(samples)))
        coeff_slices = []

        for i in np.arange(self.ntrials):
            a = pad_nextpow2(self.Epochs[i,:])
            C = pywt.wavedec(a, self.wavelet, mode='periodization', level=scales)
            Coefs, coeff_slices = pywt.coeffs_to_array(C)
            CoefMatrix[i,:] = Coefs
        self.CoefMatrix = CoefMatrix
        self.slices = coeff_slices
        return self


    def get_coef(self,wanted_scale):
        data = self.CoefMatrix

        mat = np.array([])
        for i in np.arange(self.ntrials):
            C = pywt.array_to_coeffs(data[i, :], self.slices, output_format='wavedec')
            temp = np.hstack([C[i] for i in wanted_scale])
            mat = np.hstack((mat,temp))
        return np.reshape(mat,(self.ntrials,-1))



def MaxScale(obs, wavelet):
    power = nextpow2(obs)
    return pywt.dwt_max_level(2**power, wavelet)

def get_samples(obs, scales):
    """
    Get the number of samples by frequency bands of the DWT
    :param obs: array
    :param scales: max scale
    :return:
    """
    power = nextpow2(obs)
    samples = np.zeros((scales+1,1))
    for i in np.arange(scales):
        samples[i] = (2**power)/(2**(i+1))
    samples[-1] = (2**power)/(2**(scales))
    return np.asarray(samples,dtype='int')



