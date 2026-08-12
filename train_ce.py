import argparse
import os 

import torch
import torch.optim as optim
from torchaudio.functional import resample

from tqdm import tqdm

from module.dataset import Dataset
from module.content_encoder import ContentEncoder
from module.content_features import align_content_frames, select_hubert_content
from module.audio import normalize_ssl_input
from transformers import HubertModel

parser = argparse.ArgumentParser(description="distillation of hubert")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('--hubert', default='rinna/japanese-hubert-base')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-fp16', '--fp16', action='store_true')

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    ce = ContentEncoder().to(device)
    if os.path.exists(args.content_encoder_path):
        ce.load_state_dict(torch.load(args.content_encoder_path, map_location=device, weights_only=True))
    return ce

def save_models(ce):
    print("Saving models...")
    torch.save(ce.state_dict(), args.content_encoder_path)
    print("Complete!")

device = torch.device(args.device)

CE = load_or_init_models(device)

ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)

Opt = optim.RAdam(CE.parameters(), lr=args.learning_rate)

hubert = HubertModel.from_pretrained(args.hubert).to(device).eval()

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)

        with torch.amp.autocast(device.type, enabled=args.fp16):
            with torch.no_grad():
                hubert_wave = normalize_ssl_input(resample(wave, 24000, 16000))
                h = hubert(hubert_wave, output_hidden_states=True).hidden_states
                hubert_features = select_hubert_content(h)

        Opt.zero_grad()
        with torch.amp.autocast(device.type, enabled=args.fp16):
            z = CE.encode(wave)
            hubert_features = align_content_frames(hubert_features, z.shape[2])
            loss = (z - hubert_features).abs().mean()

        scaler.scale(loss).backward()
        scaler.step(Opt)

        scaler.update()

        step_count += 1

        tqdm.write(f"Step {step_count}, loss: {loss.item()}")

        bar.update(N)

        if batch % 500 == 0:
            save_models(CE)

print("Training Complete!")
save_models(CE)
