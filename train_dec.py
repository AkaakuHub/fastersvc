import argparse
import os 

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from module.dataset import Dataset
from module.loss import MultiResolutionSTFTLoss
from module.adversarial import discriminator_loss, generator_adversarial_loss, require_finite
from module.audio import PerceptualLoudness
from module.content_encoder import ContentEncoder
from module.decoder import Decoder
from module.discriminator import Discriminator
from module.training import (
    DECODER_GRADIENT_NORM,
    DECODER_OPTIMIZER_EPSILON,
    DEFAULT_DECODER_LEARNING_RATE,
    atomic_save,
    batch_size_per_process,
    crop_aligned_batch,
    discriminator_learning_rate_at_step,
    learning_rate_at_step,
    set_optimizer_learning_rate,
    step_scaled_optimizer,
    training_data_loader,
)


parser = argparse.ArgumentParser(description="train voice conversion model")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-dip', '--discriminator-path', default='models/discriminator.pt')
parser.add_argument('-dep', '--decoder-path', default='models/decoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=DEFAULT_DECODER_LEARNING_RATE)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('--steps', default=600000, type=int)
parser.add_argument('-b', '--batch-size', default=32, type=int)
parser.add_argument('--workers', default=2 if os.name != 'nt' else 0, type=int)
parser.add_argument('--save-interval', default=100, type=int)
parser.add_argument('--log-interval', default=10, type=int)
parser.add_argument('--training-state-path', default='models/decoder-training.pt')
parser.add_argument('-fp16', '--fp16', action='store_true')

parser.add_argument('--weight-adv', default=2.5, type=float)
parser.add_argument('--weight-stft', default=1.0, type=float)
parser.add_argument('--discriminator-start-step', default=100000, type=int)

args = parser.parse_args()
if args.log_interval <= 0:
    raise ValueError("log interval must be positive")

WEIGHT_ADV = args.weight_adv
WEIGHT_STFT = args.weight_stft

def load_or_init_models(device=torch.device('cpu'), load_model_files=True):
    dec = Decoder().to(device)
    dis = Discriminator().to(device)
    if load_model_files and os.path.exists(args.decoder_path):
        dec.load_state_dict(torch.load(args.decoder_path, map_location=device, weights_only=True))
    if load_model_files and os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device, weights_only=True))
    return dec, dis


def model_state(model):
    if isinstance(model, DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def decoder_source(model, pitch):
    if isinstance(model, DistributedDataParallel):
        return model.module.generate_source(pitch)
    return model.generate_source(pitch)


def load_model_state(model, state):
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


def save_models(dec, dis, opt_dec, opt_dis, scaler, step_count):
    print("Saving models...")
    atomic_save({
        "decoder": model_state(dec),
        "discriminator": model_state(dis),
        "decoder_optimizer": opt_dec.state_dict(),
        "discriminator_optimizer": opt_dis.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step_count,
    }, args.training_state_path)
    atomic_save(model_state(dec), args.decoder_path)
    atomic_save(model_state(dis), args.discriminator_path)
    print("Complete!")


distributed = "LOCAL_RANK" in os.environ
if distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    process_rank = dist.get_rank()
    process_count = dist.get_world_size()
    device = torch.device("cuda", local_rank)
else:
    process_rank = 0
    process_count = 1
    device = torch.device(args.device)
is_primary_process = process_rank == 0

training_state_exists = os.path.exists(args.training_state_path)
Dec, Dis = load_or_init_models(device, load_model_files=not training_state_exists)
if distributed:
    Dec = DistributedDataParallel(Dec, device_ids=[device.index])
    Dis = DistributedDataParallel(Dis, device_ids=[device.index])
CE = ContentEncoder().to(device).eval()
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device, weights_only=True))

ds = Dataset(args.dataset_cache)
sampler = DistributedSampler(ds, shuffle=True) if distributed else None
process_batch_size = batch_size_per_process(args.batch_size, process_count)
dl = training_data_loader(ds, process_batch_size, args.workers, device, sampler)

scaler = torch.amp.GradScaler(device.type, enabled=args.fp16)

OptDec = optim.Adam(Dec.parameters(), lr=args.learning_rate, eps=DECODER_OPTIMIZER_EPSILON)
OptDis = optim.Adam(Dis.parameters(), lr=args.learning_rate, eps=DECODER_OPTIMIZER_EPSILON)

spectral_loss = MultiResolutionSTFTLoss().to(device)
loudness_extractor = PerceptualLoudness().to(device)

step_count = 0
if training_state_exists:
    training_state = torch.load(args.training_state_path, map_location=device, weights_only=True)
    load_model_state(Dec, training_state["decoder"])
    load_model_state(Dis, training_state["discriminator"])
    OptDec.load_state_dict(training_state["decoder_optimizer"])
    OptDis.load_state_dict(training_state["discriminator_optimizer"])
    scaler.load_state_dict(training_state["scaler"])
    step_count = training_state["step"]

epoch = 0
while step_count < args.steps:
    if sampler is not None:
        sampler.set_epoch(epoch)
    if is_primary_process:
        tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds), disable=not is_primary_process)
    for batch_data in dl:
        wave = batch_data[0]
        f0 = batch_data[1]
        N = wave.shape[0]
        
        # train generator and speaker encoder
        OptDec.zero_grad()
        with torch.amp.autocast(device.type, enabled=args.fp16):
            wave = wave.to(device, non_blocking=True)
            f0 = f0.to(device, non_blocking=True)
            wave, f0 = crop_aligned_batch(wave, f0)
            generator_learning_rate = learning_rate_at_step(args.learning_rate, step_count)
            discriminator_learning_rate = discriminator_learning_rate_at_step(
                args.learning_rate,
                step_count,
                args.discriminator_start_step,
            )
            set_optimizer_learning_rate(OptDec, generator_learning_rate)
            set_optimizer_learning_rate(OptDis, discriminator_learning_rate)

            with torch.no_grad():
                z = CE.encode(wave)
            e = loudness_extractor(wave)
            source = decoder_source(Dec, f0)
            fake = Dec(z, e, source)
            require_finite("generated waveform", fake)

            loss_stft = spectral_loss(fake, wave)
            if step_count >= args.discriminator_start_step:
                logits, _ = Dis(fake)
                loss_adv = generator_adversarial_loss(logits)
                loss_g = loss_adv * WEIGHT_ADV + loss_stft * WEIGHT_STFT
            else:
                loss_adv = fake.new_tensor(0.0)
                loss_g = loss_stft * WEIGHT_STFT
            require_finite("generator loss", loss_g)

        step_scaled_optimizer(loss_g, OptDec, scaler, Dec.parameters(), DECODER_GRADIENT_NORM)

        if step_count >= args.discriminator_start_step:
            fake = fake.detach()
            OptDis.zero_grad()
            with torch.amp.autocast(device.type, enabled=args.fp16):
                real_logits, _ = Dis(wave)
                generated_logits, _ = Dis(fake)
                loss_d = discriminator_loss(real_logits, generated_logits)
                require_finite("discriminator loss", loss_d)

            step_scaled_optimizer(loss_d, OptDis, scaler, Dis.parameters(), 1.0)
        else:
            loss_d = fake.new_tensor(0.0)

        scaler.update()

        step_count += 1
        
        if is_primary_process and (step_count == 1 or step_count % args.log_interval == 0):
            tqdm.write(f"Epoch {epoch}, Step {step_count}, Dis.: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, STFT: {loss_stft.item():.4f}")

        bar.update(N * process_count)

        if step_count % args.save_interval == 0:
            if distributed:
                dist.barrier()
            if is_primary_process:
                save_models(Dec, Dis, OptDec, OptDis, scaler, step_count)
            if distributed:
                dist.barrier()
        if step_count >= args.steps:
            break
    epoch += 1

if is_primary_process:
    print("Training Complete!")
    if step_count % args.save_interval != 0:
        save_models(Dec, Dis, OptDec, OptDis, scaler, step_count)
if distributed:
    dist.destroy_process_group()
