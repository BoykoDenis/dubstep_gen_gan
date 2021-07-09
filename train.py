
import os
import torch
import torch.nn as nn
import torchaudio
#import AudioConverter
import matplotlib.pyplot as plt
from Generator import Generator

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

epochs = 10



data_path = os.path.join("data", "sample3.wav")
save_path = os.path.join("saved", "sample3changed.wav")
#print(data_path)

#print(str(torchaudio.get_audio_backend()))
waveform, sample_rate = torchaudio.load( data_path ) #supports only wav files P.S.: Waveform can be to chanaled --> 2 dim tensor
print(waveform.shape)
print(sample_rate)

model = Generator(2, 1, 1)


plt.plot(waveform[0][:10000])
plt.plot(waveform[1][:10000])
plt.show()



waveform = waveform[:][:10000].unsqueeze(0)



waveform = waveform.permute(2, 0, 1)
print(type(waveform))
output = model(waveform[:][:1000000])
output = output[0].detach()
print(output.shape)
output = output.squeeze(1).squeeze(1)
plt.plot(output)
torchaudio.save(save_path, output, 44100)
#plt.plot(output[1])
plt.show()


'''
for track in os.listdir("data\\"):

    for epoch in epochs:
        pass
'''



#playsound( data_path + "sample 1.mp3")