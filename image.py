import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Imagefourier:
    def __init__(self, image):
        self.image=cv2.imread(image,0)
        self.image=cv2.resize(self.image,(400,400))
        self.fftofimage=np.fft.fftshift(np.fft.fft2(self.image))
        self.magnitude=np.sqrt(np.real(self.fftofimage) ** 2 + np.imag(self.fftofimage) ** 2)
        self.magnitude_spectrum = 20*np.log(np.abs(self.fftofimage))
        self.phase=np.arctan2(np.imag(self.fftofimage), np.real(self.fftofimage))
        self.phase_spectrum = np.angle(self.fftofimage)
        


class Processing(Imagefourier):

  def __init__(self,image):
    Imagefourier.__init__(self,image)
  
  def combine(self, magnitude, phase):
    self.outputcombine = np.multiply(magnitude, np.exp(1j *phase))
    self.img_output = np.real(np.fft.ifft2(self.outputcombine))
