import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
import urllib.request
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

import zipfile
import urllib.request

print("Preparing WavCaps SoundBible subset...")
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
zip_path = os.path.join(data_dir, "SoundBible.zip")
json_path = os.path.join(data_dir, "sb_final.json")
extract_dir = os.path.join(data_dir, "extracted")
audio_dir = os.path.join(extract_dir, "mnt", "fast", "nobackup", "scratch4weeks", "xm00178", "WavCaps", "data", "waveforms", "SoundBible_flac")

if os.path.exists("SoundBible.zip") and not os.path.exists(zip_path):
    import shutil
    shutil.move("SoundBible.zip", zip_path)
if os.path.exists("sb_final.json") and not os.path.exists(json_path):
    import shutil
    shutil.move("sb_final.json", json_path)

if not os.path.exists(zip_path):
    print("Downloading SoundBible.zip...")
    urllib.request.urlretrieve("https://huggingface.co/datasets/cvssp/WavCaps/resolve/main/Zip_files/SoundBible/SoundBible.zip", zip_path)

if not os.path.exists(json_path):
    print("Downloading sb_final.json...")
    urllib.request.urlretrieve("https://huggingface.co/datasets/cvssp/WavCaps/resolve/main/json_files/SoundBible/sb_final.json", json_path)

if not os.path.exists(audio_dir):
    print("Extracting audio...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

with open(json_path, 'r') as f:
    sb_json = json.load(f)
    
dataset_list = sb_json['data']
samples = []
for item in dataset_list:
    flac_path = os.path.join(audio_dir, f"{item['id']}.flac")
    if os.path.exists(flac_path):
        samples.append({
            'audio_path': flac_path,
            'caption': item['caption'],
            'id': item['id'],
            'title': item.get('title', 'Audio ' + str(item['id']))
        })

print(f"Loaded {len(samples)} actual audio samples.")
total_samples = len(samples)
samples = samples[:total_samples]

train_samples = samples[:int(total_samples*0.8)]
val_samples = samples[int(total_samples*0.8):]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class WavCapsSubsetDataset(Dataset):
    def __init__(self, samples, max_len=32):
        self.samples = samples
        self.max_len = max_len
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        audio_path = item['audio_path']
        import soundfile as sf
        waveform_np, sr = sf.read(audio_path)
        waveform = torch.tensor(waveform_np, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t() # shape to (channels, samples)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        audio_tensor = waveform
            
        max_audio_len = 48000
        if audio_tensor.shape[1] > max_audio_len:
            audio_tensor = audio_tensor[:, :max_audio_len]
        else:
            pad_amount = max_audio_len - audio_tensor.shape[1]
            audio_tensor = F.pad(audio_tensor, (0, pad_amount))
            
        mel_spec = self.mel_transform(audio_tensor)
        log_mel = self.amplitude_to_db(mel_spec)
        
        caption = item['caption']
        text_inputs = tokenizer(caption, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        
        return {
            'mel': log_mel,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'caption': caption
        }

train_dataset = WavCapsSubsetDataset(train_samples)
val_dataset = WavCapsSubsetDataset(val_samples)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 16 * 23, embed_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token)

class DualEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.audio_enc = AudioEncoder(embed_dim)
        self.text_enc = TextEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, mel, input_ids, attention_mask):
        audio_emb = self.audio_enc(mel)
        text_emb = self.text_enc(input_ids, attention_mask)
        
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        return audio_emb, text_emb, self.logit_scale.exp()

model = DualEncoder().to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

def clip_loss(audio_emb, text_emb, logit_scale):
    logits_per_audio = logit_scale * audio_emb @ text_emb.T
    logits_per_text = logits_per_audio.T
    
    labels = torch.arange(len(logits_per_audio), dtype=torch.long, device=logits_per_audio.device)
    loss_a = F.cross_entropy(logits_per_audio, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_a + loss_t) / 2

if __name__ == '__main__':
    model.train()
    # Only 1 batch for quick test
    for batch in train_loader:
        mel = batch['mel'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        audio_emb, text_emb, logit_scale = model(mel, input_ids, attention_mask)
        loss = clip_loss(audio_emb, text_emb, logit_scale)
        
        loss.backward()
        optimizer.step()
        break
    print("Training step passed.")

@torch.no_grad()
def evaluate_retrieval(model, dataloader):
    model.eval()
    all_audio_emb = []
    all_text_emb = []
    
    for batch in dataloader:
        mel = batch['mel'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        a_emb, t_emb, _ = model(mel, input_ids, attention_mask)
        all_audio_emb.append(a_emb.cpu())
        all_text_emb.append(t_emb.cpu())
        break # Test one batch
        
    all_audio_emb = torch.cat(all_audio_emb, dim=0).numpy()
    all_text_emb = torch.cat(all_text_emb, dim=0).numpy()
    
    distances = pairwise_distances(all_audio_emb, all_text_emb, metric='cosine')
    return distances

if __name__ == '__main__':
    dists = evaluate_retrieval(model, val_loader)
    print("Eval shapes:", dists.shape)
    print("All passed!")
