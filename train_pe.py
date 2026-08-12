import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from module.dataset import Dataset
from module.pitch_estimator import PitchEstimator
from module.training import atomic_save, step_scaled_optimizer


parser = argparse.ArgumentParser(description="distill WORLD pitch estimation")
parser.add_argument("--dataset-cache", default="dataset_cache")
parser.add_argument("--pitch-estimator-path", default="models/pitch_estimator.pt")
parser.add_argument("--training-state-path", default="models/pitch-estimator-training.pt")
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--device", default="cuda")
parser.add_argument("--steps", default=10000, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--save-interval", default=500, type=int)
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()


def save_training_state(model, optimizer, scaler, step_count):
    atomic_save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step_count,
    }, args.training_state_path)
    atomic_save(model.state_dict(), args.pitch_estimator_path)


device = torch.device(args.device)
model = PitchEstimator().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)
step_count = 0

if os.path.exists(args.training_state_path):
    training_state = torch.load(args.training_state_path, map_location=device, weights_only=True)
    model.load_state_dict(training_state["model"])
    optimizer.load_state_dict(training_state["optimizer"])
    scaler.load_state_dict(training_state["scaler"])
    step_count = training_state["step"]
elif os.path.exists(args.pitch_estimator_path):
    model.load_state_dict(torch.load(args.pitch_estimator_path, map_location=device, weights_only=True))

dataset = Dataset(args.dataset_cache)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
loss_function = nn.CrossEntropyLoss().to(device)

epoch = 0
while step_count < args.steps:
    tqdm.write(f"Epoch #{epoch}")
    progress = tqdm(total=len(dataset))
    for waveforms, pitch, _ in loader:
        waveforms = waveforms.to(device)
        pitch = pitch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=args.fp16):
            logits = model.logits(waveforms)
            labels = model.freq2id(pitch.squeeze(1))
            loss = loss_function(logits, labels)

        step_scaled_optimizer(loss, optimizer, scaler, model.parameters(), 1.0)
        scaler.update()
        step_count += 1
        tqdm.write(f"Epoch {epoch}, Step {step_count}, loss: {loss.item():.6f}")
        progress.update(waveforms.shape[0])

        if step_count % args.save_interval == 0:
            save_training_state(model, optimizer, scaler, step_count)
        if step_count >= args.steps:
            break
    epoch += 1

save_training_state(model, optimizer, scaler, step_count)
