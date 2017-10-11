import lws

import scipy
import numpy as np
import librosa
import scipy


y, sr = librosa.load("6.1.wav",sr=16000)
#_hop_length=200;_win_length=800;_fftsize=2048;_n_fft=2048
#_hop_length=128;_win_length=512;_fftsize=2048;_n_fft=2048
_hop_length=128;_win_length=512;_fftsize=512;
#_hop_length=256;_win_length=1024;_fftsize=1024;
window = scipy.signal.hanning
#stft_dtype = np.float64
stft_dtype = np.complex128
'''
lws_processor=lws.lws(_win_length, _hop_length, mode="speech",batch_iterations = 100,stft_opts={'perfectrec':True,'fftsize':_fftsize})
#X = lws_processor.stft(y)
X = librosa.stft(y, window=window,center=True,n_fft=_fftsize, hop_length=_hop_length, win_length=_win_length, dtype=stft_dtype).T
print("X: ",X.shape)
X0 = np.abs(X) # Magnitude spectrogram
X1 = lws_processor.run_lws(X0)
print("X1 ",X1.shape)
#X2 = lws_processor.istft(X1).astype(np.float32)
X2 = librosa.istft(X1.T,hop_length=_hop_length, win_length=_win_length)
#_,X2 = scipy.signal.istft(Zxx=X1.T, fs=sr,nfft=_n_fft,nperseg=_win_length,noverlap=_hop_length)
#X2=X2.astype(np.float32)
librosa.output.write_wav("istft.wav", X2, sr)
'''
lws_processor=lws.lws(_win_length, _hop_length, mode="speech",batch_iterations = 100,stft_opts={'perfectrec':True,'fftsize':_fftsize})
X_1 = librosa.stft(y, window=window,center=True,n_fft=_fftsize, hop_length=_hop_length, win_length=_win_length, dtype=stft_dtype).T
X = lws_processor.stft(y)
X0 = np.abs(X_1) # Magnitude spectrogram
X1 = lws_processor.run_lws(X0)

print("X_1.shape", X_1.shape)
print("X.shape", X.shape)
print(type(X))
print(type(X[0][10]))
print(type(X1[0][10]))
print(np.max(np.abs(np.abs(X)-X0)))

X2_1 = librosa.istft(X_1.T,hop_length=_hop_length, win_length=_win_length, center=False)
X2 = librosa.istft(X.T,hop_length=_hop_length, win_length=_win_length, center=False)
X2_2 = librosa.istft(X0.T,hop_length=_hop_length, win_length=_win_length, center=False)
X2_3 = lws_processor.istft(X1).astype(np.float32)
librosa.output.write_wav("istft_1.wav", X2_1, sr)
librosa.output.write_wav("istft.wav", X2, sr)
librosa.output.write_wav("istft_2.wav", X2_2, sr)
librosa.output.write_wav("istft_3.wav", X2_3, sr)
