import torch
import torchaudio
from Generator import Generator as gen

song_lenght = 2 # in minutes
song_sample_rate = 44100
model_path = "models\\"
model_name = "1rnn221v1.pt"
save_path = "saved\\"


def load_model_params(model_path, model):
    checkpoint = torch.load()
    model.load_state_dict(checkpoint["state_dictionary"])
    return model

model = gen(2, 2, 1)
model = load_model_params(model_path + model_name)
song_step = torch.randn(1, 1, 2)
output, h_save = model(song_step)
final_song = []

for song_l in range(song_lenght * song_sample_rate):
    song_step, h_save = model(song_step, h_save)
    final_song.append(song_step)

final_song = torch.cat(final_song)
torchaudio.save(save_path, output, song_sample_rate)

