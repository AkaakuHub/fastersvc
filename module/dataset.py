import torch
import torchaudio
from pathlib import Path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_path = 'dataset_cache'):
        super().__init__()
        self.dir_path = Path(dir_path)
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
        wf, _ = torchaudio.load(self.dir_path / f"{item_id}.wav")
        wf = wf.mean(dim=0)
        return wf, f0, spk_id
