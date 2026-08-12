import argparse
from pathlib import Path
import torch

from module.dataset import Dataset
from module.content_encoder import ContentEncoder
from module.content_features import encode_index_features

def shuffle(tensor, dim):
    indices = torch.randperm(tensor.size(dim), device=tensor.device)
    shuffled_tensor = tensor.index_select(dim, indices)
    return shuffled_tensor

parser = argparse.ArgumentParser(description="extract index")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-size', default=1024, type=int)
parser.add_argument('--stride', default=4, type=int)
parser.add_argument('-o', '--output', default='models/index.pt')
parser.add_argument('-d', '--device', default='cpu')

args = parser.parse_args()

device = torch.device(args.device)
CE = ContentEncoder().to(device).eval()
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device, weights_only=True))

features = []
total_length = 0

ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

print("Extracting...")
for wave, _, _ in dl:
    feat = encode_index_features(CE, wave.to(device), args.stride)
    total_length += feat.shape[2]
    features.append(feat)
    if total_length >= args.size:
        break

features = torch.cat(features, dim=2)
idx = shuffle(features, dim=2)[:, :, :args.size]
print(f"Extracted {idx.shape[2]} vectors.")

print("Saving...")
output_parent = Path(args.output).parent
output_parent.mkdir(parents=True, exist_ok=True)
torch.save(idx, args.output)

print("Complete.")
