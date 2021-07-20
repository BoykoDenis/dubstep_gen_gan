import torch
import torchaudio
from Generator import Generator as gen

song_lenght = 30 # in seconds
song_sample_rate = 44100
model_path = "models\\"
model_name = "1rnn221v1.pt"
save_path = "saved\\"
song_name = "test1.wav"

device = torch.device('cuda')

def load_model_params(model_path, model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dictionary"])
    return model

model = gen(2, 2, 1).to(device)
model = load_model_params(model_path + model_name, model)
song_step = torch.randn(1, 1, 2).to(device)
song_step, h_save = model.forward(song_step)

final_song = []
sample_lenght = song_lenght * song_sample_rate
for song_l in range(sample_lenght):

    song_step, h_save = model.running_forward(song_step, h_save)
    final_song.append(song_step)
    print(f"{(round(song_l/sample_lenght, 1))*100}%         ", end="\r")

final_song = torch.cat(final_song).squeeze(1).permute(1, 0).cpu().detach()

print(final_song.shape)
torchaudio.save(save_path+song_name, final_song, song_sample_rate)

