import torch
import torchaudio
from pathlib import Path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_path='dataset_cache', sample_rate=24000, frame_size=480):
        super().__init__()
        self.dir_path = Path(dir_path)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        wave_ids = {path.stem for path in self.dir_path.glob("*.wav")}
        pitch_ids = {path.stem for path in self.dir_path.glob("*.pt")}
        if wave_ids != pitch_ids:
            raise ValueError("dataset cache must contain one pitch file for every waveform")
        self.ids = sorted(wave_ids, key=int)
        if not self.ids:
            raise ValueError("dataset cache is empty")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id = self.ids[idx]
        f0, spk_id = torch.load(self.dir_path / f"{item_id}.pt", weights_only=True)
        wf, sample_rate = torchaudio.load(self.dir_path / f"{item_id}.wav")
        if sample_rate != self.sample_rate:
            raise ValueError(f"dataset waveform sample rate must be {self.sample_rate}")
        wf = wf.mean(dim=0)
        if wf.shape[0] != f0.shape[-1] * self.frame_size:
            raise ValueError("dataset waveform and pitch frame counts differ")
        return wf, f0, spk_id
