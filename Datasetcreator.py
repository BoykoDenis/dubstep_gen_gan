from os import path
from pydub import AudioSegment


data = AudioSegment.from_mp3(src)
data.export(dst, format = "wav")
