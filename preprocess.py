import argparse
from pathlib import Path
import json

import torch
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.common import compute_f0
from module.preprocessing import split_complete_segments, trim_silence


parser = argparse.ArgumentParser(description="preprocess")

parser.add_argument('input')
parser.add_argument('-o', '--output', default='dataset_cache')
parser.add_argument('-len', '--length', default=24000, type=int)
parser.add_argument('--num-speakers', default=8192, type=int)
parser.add_argument('-m', '--max-files', default=-1, type=int)
parser.add_argument('--pitch-algorithm', default='harvest', choices=['harvest', 'dio'])
parser.add_argument('--trim-top-db', default=30.0, type=float)
parser.add_argument('--speaker-infomation', default='speaker_infomation.json')

args = parser.parse_args()

input_parent = Path(args.input)
dataset_files = []

support_exts = ['mp3', 'wav', 'ogg']
for e in support_exts:
    dataset_files += list(input_parent.glob(f"**/*.{e}"))
dataset_files.sort()
if args.max_files != -1:
    dataset_files = dataset_files[:args.max_files]

# create output directory
output_parent = Path(args.output)
output_parent.mkdir(parents=True, exist_ok=True)

parent_paths = []
counter = 0
for path in tqdm(dataset_files):
    tqdm.write(f"processing {str(path)}")
    parent_path = path.parent
    wf, sr = torchaudio.load(path)
    wf = wf.mean(dim=0, keepdim=True)
    wf = resample(wf, sr, 24000)
    wf = trim_silence(wf, top_db=args.trim_top_db)
    chunks = split_complete_segments(wf, args.length)
    for chunk in chunks:
        # f0
        f0 = compute_f0(chunk, algorithm=args.pitch_algorithm)


        # get spekaer id
        if parent_path not in parent_paths:
            parent_paths.append(parent_path)
        spk_id = parent_paths.index(parent_path)
        if spk_id >= args.num_speakers:
            raise ValueError(f"speaker count exceeds --num-speakers={args.num_speakers}")

        # save
        output_pt_path = output_parent / f"{counter}.pt"
        torch.save((f0[0].detach().cpu(), spk_id), output_pt_path)
        output_wave_path = output_parent / f"{counter}.wav"
        torchaudio.save(output_wave_path, src=chunk, sample_rate=24000)
        counter += 1

# output speaker details
output_obj = {}
for i, p in enumerate(parent_paths):
    output_obj[str(i)] = str(p)

with open(args.speaker_infomation, 'w') as f:
    json.dump(output_obj, f)

print("complete!")
