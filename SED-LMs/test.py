import os
os.environ["CUDA_VISIBLE_DEVICES"] ='2'
os.environ['CURL_CA_BUNDLE'] = ''
import librosa
import torch
import torch.nn.functional as F
from models.bert_captioning import BertCaptionModel
from models.bart_captioning import BartCaptionModel

checkpoint_path = ""
audio_path = "DCASE/waveforms/val/Y--4gqARaEJE.flac"
cp = torch.load(checkpoint_path)
config = cp["config"]

model = BartCaptionModel(config)
model.load_state_dict(cp["model"],strict =False)
device = torch.device(config["device"])
model.to(device)

waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
waveform = torch.tensor(waveform)
max_length = 16000 * 10
if len(waveform) > max_length:
    waveform = waveform[:max_length]
else:
    waveform = F.pad(waveform, [0, max_length - len(waveform)], "constant", 0.0)
waveform = waveform.unsqueeze(0)

model.eval()
with torch.no_grad():
    waveform = waveform.to(device)
    caption = model.generate(samples=waveform)
print(caption)


