import os
import torch
import torchaudio
import AudioConverter
import matplotlib.pyplot as plt
#import librosa

#from playsound import playsound

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

data_path = "data\\sample3.wav"
#data_path = os.path.join("data", "sample2.mp3")
#print(data_path)

#print(str(torchaudio.get_audio_backend()))
waveform, sample_rate = torchaudio.load( data_path ) #supports only wav files P.S.: Waveform can be to chanaled --> 2 dim tensor
print(waveform.shape)
print(sample_rate)
plt.plot(waveform[0][:10000])
plt.show()

#playsound( data_path + "sample 1.mp3")