import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from module.common import compute_f0


def publish_cache_item(source_wave, source_pitch, output_directory, pitch):
    output_wave = output_directory / source_wave.name
    output_pitch = output_directory / source_pitch.name
    if not output_wave.exists():
        os.link(source_wave, output_wave)
    elif not os.path.samefile(source_wave, output_wave):
        raise ValueError(f"output waveform differs from source: {output_wave}")

    _, speaker_id = torch.load(source_pitch, weights_only=True)
    temporary_pitch = output_pitch.with_suffix(".pt.saving")
    torch.save((pitch.detach().cpu(), speaker_id), temporary_pitch)
    os.replace(temporary_pitch, output_pitch)


def rebuild_cache_item(task):
    source_directory, output_directory, item_id, pitch_algorithm = task
    source_wave = source_directory / f"{item_id}.wav"
    source_pitch = source_directory / f"{item_id}.pt"
    output_pitch = output_directory / f"{item_id}.pt"
    output_wave = output_directory / f"{item_id}.wav"
    if output_pitch.exists() and output_wave.exists():
        return item_id

    waveform, _ = torchaudio.load(source_wave)
    waveform = waveform.mean(dim=0, keepdim=True)
    pitch = compute_f0(waveform, algorithm=pitch_algorithm)[0]
    publish_cache_item(source_wave, source_pitch, output_directory, pitch)
    return item_id


def main():
    parser = argparse.ArgumentParser(description="rebuild cached pitch at the current frame timing")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--pitch-algorithm", default="harvest", choices=["harvest", "dio"])
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()

    source_directory = Path(args.source).resolve()
    output_directory = Path(args.output).resolve()
    if source_directory == output_directory:
        raise ValueError("source and output cache directories must differ")
    output_directory.mkdir(parents=True, exist_ok=True)

    wave_ids = {path.stem for path in source_directory.glob("*.wav")}
    pitch_ids = {path.stem for path in source_directory.glob("*.pt")}
    if wave_ids != pitch_ids or not wave_ids:
        raise ValueError("source cache must contain matching waveform and pitch files")
    item_ids = sorted(wave_ids, key=int)
    tasks = [
        (source_directory, output_directory, item_id, args.pitch_algorithm)
        for item_id in item_ids
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _ in tqdm(executor.map(rebuild_cache_item, tasks), total=len(tasks)):
            pass


if __name__ == "__main__":
    main()
