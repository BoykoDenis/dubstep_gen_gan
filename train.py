
import os
from typing import Sequence
import torch
import torch.nn as nn
import torchaudio
#import AudioConverter
import matplotlib.pyplot as plt
from Generator import Generator

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

device = torch.device("cuda")

epochs = 4
lr = 0.001
seq_lenght = 88200
model_save_path = 'models\\'
model_name = '1rnn221v1.pt' # 1 - number of layers of rnn; rnn - type of layer; 221 -> (input size, hidden_size, number of layers; v - version)

#input --> nn --> output

# input*wn--> f(sum) if > barier than pass

def save_mod(state, path, name):

	torch.save(state, path+name)


def load_Data(data_path):
    waveform, sample_rate = torchaudio.load( data_path ) #supports only wav files P.S.: Waveform can be to chanaled --> 2 dim tensor
    #print(waveform.shape)
    #print(sample_rate)
    return waveform, sample_rate


def iterator(waveform):
    pass


def convert_data(data):
    data = data.unsqueeze(0)
    data = data.permute(2, 0, 1)
    return data


def unconvert_data():
    pass


#waveform = waveform[:][:10000].unsqueeze(0)


'''
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


model = Generator(2, 2, 1).to(device)
params = model.parameters()
optimizer = torch.optim.Adam(params, lr = lr)
loss_function = torch.nn.MSELoss(reduction = 'sum')



#for track in os.listdir("data\\"):

try:
    checkpoint = torch.load(model_save_path + model_name)
    model.load_state_dict(checkpoint["state_dictionary"])

except:
    print('load_error', end='\n')

for epoch in range(epochs):

    for song in range(1, 5):
        data_path = os.path.join("data", f"sample ({song}).wav")
        waveform, sample_rate = load_Data(data_path)
        song_lenght = waveform.shape[1] // seq_lenght
        checkpoint = {"state_dictionary" : model.state_dict()}
        save_mod(checkpoint, model_save_path, model_name)
        running_loss = 0.0

        for idx in range(song_lenght):
            try:
                data = convert_data(waveform[:, idx*seq_lenght:(idx+1)*seq_lenght]).to(device) # takes the part of a song as a sequence
                label = convert_data(waveform[:, idx+1:idx+1+seq_lenght]).to(device) # takes the part of a song as a sequence the same lenght as data but 1 sample further

                output, _ = model.forward(data)
                loss = loss_function(output[0], label)
                #print(output[0].shape)
                #print(output.shape)
                #plt.plot(output)

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"epoch: {epoch}, dataset progress: {idx}/{song_lenght}, loss: {running_loss/(idx+1)}", end = '\r')
                #a = input() # stoper for testing, remove before training
            except Exception as e:
                print(e, end = '\n')
                continue


