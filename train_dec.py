import argparse
import os 

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from module.dataset import Dataset
from module.loss import MultiResolutionSTFTLoss
from module.adversarial import discriminator_loss, generator_adversarial_loss, require_finite
from module.content_encoder import ContentEncoder
from module.decoder import Decoder
from module.common import energy
from module.discriminator import Discriminator


parser = argparse.ArgumentParser(description="train voice conversion model")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-dip', '--discriminator-path', default='models/discriminator.pt')
parser.add_argument('-dep', '--decoder-path', default='models/decoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('--steps', default=600000, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--save-interval', default=100, type=int)
parser.add_argument('--training-state-path', default='models/decoder-training.pt')
parser.add_argument('-fp16', '--fp16', action='store_true')

parser.add_argument('--weight-adv', default=1.0, type=float)
parser.add_argument('--weight-stft', default=1.0, type=float)

args = parser.parse_args()

WEIGHT_ADV = args.weight_adv
WEIGHT_STFT = args.weight_stft

def load_or_init_models(device=torch.device('cpu')):
    dec = Decoder().to(device)
    dis = Discriminator().to(device)
    if os.path.exists(args.decoder_path):
        dec.load_state_dict(torch.load(args.decoder_path, map_location=device, weights_only=True))
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device, weights_only=True))
    return dec, dis


def atomic_save(value, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary_path = f"{path}.saving"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def save_models(dec, dis, opt_dec, opt_dis, scaler, step_count):
    print("Saving models...")
    atomic_save({
        "decoder": dec.state_dict(),
        "discriminator": dis.state_dict(),
        "decoder_optimizer": opt_dec.state_dict(),
        "discriminator_optimizer": opt_dis.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step_count,
    }, args.training_state_path)
    atomic_save(dec.state_dict(), args.decoder_path)
    atomic_save(dis.state_dict(), args.discriminator_path)
    print("Complete!")


def center(wave, length=16000):
    c = wave.shape[1] // 2
    half_len = length // 2
    return wave[:, c-half_len:c+half_len]


device = torch.device(args.device)

Dec, Dis = load_or_init_models(device)
CE = ContentEncoder().to(device).eval()
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device, weights_only=True))

ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)

OptDec = optim.AdamW(Dec.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptDis = optim.AdamW(Dis.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))

spectral_loss = MultiResolutionSTFTLoss().to(device)

step_count = 0
if os.path.exists(args.training_state_path):
    training_state = torch.load(args.training_state_path, map_location=device, weights_only=True)
    Dec.load_state_dict(training_state["decoder"])
    Dis.load_state_dict(training_state["discriminator"])
    OptDec.load_state_dict(training_state["decoder_optimizer"])
    OptDis.load_state_dict(training_state["discriminator_optimizer"])
    scaler.load_state_dict(training_state["scaler"])
    step_count = training_state["step"]

epoch = 0
while step_count < args.steps:
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        N = wave.shape[0]
        
        # train generator and speaker encoder
        OptDec.zero_grad()
        with torch.amp.autocast(device.type, enabled=args.fp16):
            wave = wave.to(device)
            f0 = f0.to(device)

            with torch.no_grad():
                z = CE.encode(wave)
            e = energy(wave)
            fake = Dec.synthesize(z, f0, e)
            require_finite("generated waveform", fake)

            logits, _ = Dis(center(fake))
            loss_adv = generator_adversarial_loss(logits)

            loss_stft = spectral_loss(fake, wave)
            loss_g = loss_adv * WEIGHT_ADV + loss_stft * WEIGHT_STFT
            require_finite("generator loss", loss_g)

        scaler.scale(loss_g).backward()
        nn.utils.clip_grad_norm_(Dec.parameters(), 1.0)
        scaler.step(OptDec)

        # train discriminator
        fake = fake.detach()
        OptDis.zero_grad()
        with torch.amp.autocast(device.type, enabled=args.fp16):
            real_logits, _ = Dis(center(wave))
            generated_logits, _ = Dis(center(fake))
            loss_d = discriminator_loss(real_logits, generated_logits)
            require_finite("discriminator loss", loss_d)

        scaler.scale(loss_d).backward()
        nn.utils.clip_grad_norm_(Dis.parameters(), 1.0)
        scaler.step(OptDis)

        scaler.update()

        step_count += 1
        
        tqdm.write(f"Epoch {epoch}, Step {step_count}, Dis.: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, STFT: {loss_stft.item():.4f}")

        bar.update(N)

        if step_count % args.save_interval == 0:
            save_models(Dec, Dis, OptDec, OptDis, scaler, step_count)
        if step_count >= args.steps:
            break
    epoch += 1

print("Training Complete!")
save_models(Dec, Dis, OptDec, OptDis, scaler, step_count)
