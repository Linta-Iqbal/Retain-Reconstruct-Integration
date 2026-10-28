"""CMU-MOSI pickle dataset adapter used by the final notebook."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


DEFAULT_IGNORED_SEGMENT_IDS = {"5W7Z1C_fDaE_24", "v0zCBqDeKcE_16"}


class MOSIDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        visual_dim: int = 20,
        audio_dim: int = 5,
        ignored_segment_ids: Iterable[str] = DEFAULT_IGNORED_SEGMENT_IDS,
    ) -> None:
        with Path(path).open("rb") as handle:
            samples = pickle.load(handle)
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        ignored = set(ignored_segment_ids)
        self.samples: list[Any] = []
        self.ignored_ids: list[str] = []

        for sample in samples:
            features, _, raw_id = sample
            sample_id = str(raw_id)
            visual = np.asarray(features[1])
            audio = np.asarray(features[2])
            visual_length, audio_length = int(features[4]), int(features[5])
            missing_visual = (
                sample_id in ignored
                or visual_length <= 0
                or visual.ndim == 0
                or visual.shape[0] == 0
            )
            if missing_visual:
                self.ignored_ids.append(sample_id)
                continue
            if audio_length <= 0 or audio.ndim == 0 or audio.shape[0] == 0:
                raise ValueError(f"{sample_id}: missing audio stream")
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        features, target, raw_id = self.samples[index]
        visual = np.asarray(features[1], dtype=np.float32)
        audio = np.asarray(features[2], dtype=np.float32)
        words = list(features[3])
        visual_length, audio_length = int(features[4]), int(features[5])
        if visual.shape != (visual_length, self.visual_dim):
            raise ValueError(
                f"{raw_id}: expected visual [{visual_length},{self.visual_dim}], got {visual.shape}"
            )
        if audio.shape != (audio_length, self.audio_dim):
            raise ValueError(
                f"{raw_id}: expected audio [{audio_length},{self.audio_dim}], got {audio.shape}"
            )
        label = float(np.asarray(target).reshape(-1)[0])
        return (
            torch.as_tensor(visual),
            torch.as_tensor(audio),
            words,
            visual_length,
            audio_length,
            torch.tensor(label, dtype=torch.float32),
            str(raw_id),
            index,
        )


def collate_mosi(batch) -> dict[str, Any]:
    visual, audio, words, visual_lengths, audio_lengths, labels, ids, indices = zip(*batch)
    return {
        "visual": pad_sequence(visual, batch_first=True),
        "audio": pad_sequence(audio, batch_first=True),
        "visual_lengths": torch.tensor(visual_lengths, dtype=torch.long),
        "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
        "words": list(words),
        "labels": torch.stack(labels),
        "ids": list(ids),
        "indices": torch.tensor(indices, dtype=torch.long),
    }

