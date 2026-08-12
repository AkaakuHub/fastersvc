import argparse
import os 

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from module.dataset import Dataset
from module.pitch_estimator import PitchEstimator

parser = argparse.ArgumentParser(description="train pitch estimation")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('-pep', '--pitch_estimator_path', default='models/pitch_estimator.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-len', '--length', default=48000, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', '--fp16', action='store_true')
parser.add_argument('--algorithm', default='harvest', type=str, choices=['harvest', 'dio'])

args = parser.parse_args()


def load_or_init_models(device=torch.device('cpu')):
    pe = PitchEstimator().to(device)
    if os.path.exists(args.pitch_estimator_path):
        pe.load_state_dict(torch.load(args.pitch_estimator_path, map_location=device, weights_only=True))
    return pe

def save_models(pe):
    print("Saving models...")
    torch.save(pe.state_dict(), args.pitch_estimator_path)
    print("Complete!")


device = torch.device(args.device)
PE = load_or_init_models(device)

ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)

Opt = optim.AdamW(PE.parameters(), lr=args.learning_rate)

weight = torch.ones(PE.output_channels)
weight[0] = 0.02
CrossEntropy = nn.CrossEntropyLoss(weight).to(device)

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device) * torch.rand(N, 1, device=device) * 2
        back_voice = wave.roll(1, dims=0)
        noise_gain = torch.rand(wave.shape[0], 1, device=device)
        back_voice_gain = torch.rand(wave.shape[0], 1, device=device) * 0.5
        noise = torch.randn_like(wave)
        wave = wave + noise_gain * noise + back_voice_gain * back_voice
        f0 = f0.to(device)

        Opt.zero_grad()
        with torch.amp.autocast(device.type, enabled=args.fp16):
            logits = PE.logits(wave)
            label = PE.freq2id(f0.squeeze(1))
            loss = CrossEntropy(logits, label)

        scaler.scale(loss).backward()
        scaler.step(Opt)

        scaler.update()

        step_count += 1

        tqdm.write(f"Step {step_count}, loss: {loss.item()}")

        bar.update(N)

        if batch % 1000 == 0:
            save_models(PE)

print("Training Complete!")
save_models(PE)
