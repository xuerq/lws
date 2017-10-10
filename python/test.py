import matplotlib
import lws
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
y, sr = librosa.load("6.1.wav",sr=16000)
'''
print("y.shape ",y.shape)
print (y)
D = librosa.stft(y,hop_length=200, win_length=800)
print(D.shape)
librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time') 
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB') 
plt.tight_layout() 
y_hat = librosa.istft(D,hop_length=200, win_length=800)
#y_hat = librosa.istft(D)
print("y_hat.shape ",y_hat.shape)
print(y_hat)

librosa.output.write_wav("istft.wav", y_hat, sr)
'''
#lws_processor=lws.lws(512,128, mode="speech") # 512: window length; 128: window shift
lws_processor=lws.lws(800,200, mode="speech") # 512: window length; 128: window shift
#X = lws_processor.stft(y) # where x is a single-channel waveform
X = librosa.stft(y, n_fft=800, hop_length=200, win_length=800, dtype=np.complex128).T
print(X.shape)
X0 = np.abs(X) # Magnitude spectrogram
X1 = lws_processor.run_lws(X0) # reconstruction from magnitude (in general, one can reconstruct from an initial complex spectrogram)
#X2 = lws_processor.istft(X1) # where x is a single-channel waveform
X2 = librosa.istft(X1.T,hop_length=200, win_length=800)
librosa.output.write_wav("istft.wav", X2, sr)


