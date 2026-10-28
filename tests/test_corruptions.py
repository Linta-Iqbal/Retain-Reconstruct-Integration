import numpy as np
import torch
from PIL import Image

from rri.corruptions import (
    gaussian_image_noise,
    gaussian_sequence_noise,
    mask_words,
    salt_and_pepper,
)


def test_image_corruptions_are_seed_reproducible():
    image = Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8))
    first = gaussian_image_noise(image, 0.3, np.random.default_rng(7))
    second = gaussian_image_noise(image, 0.3, np.random.default_rng(7))
    assert np.array_equal(np.asarray(first), np.asarray(second))

    corrupted = salt_and_pepper(image, 0.25, np.random.default_rng(9))
    changed_pixels = np.any(np.asarray(corrupted) != 127, axis=-1).sum()
    assert changed_pixels == 16


def test_word_masking_uses_requested_number_of_words():
    result = mask_words(
        "one two three four", 0.5, np.random.default_rng(2), replacement_token="MASK"
    )
    assert result.split().count("MASK") == 2


def test_sequence_corruption_does_not_touch_padding():
    values = torch.ones(2, 4, 3)
    values[0, 2:] = 99.0
    lengths = torch.tensor([2, 4])
    first = gaussian_sequence_noise(values, lengths, 0.5, seed=10, sample_indices=[3, 4])
    second = gaussian_sequence_noise(values, lengths, 0.5, seed=10, sample_indices=[3, 4])
    assert torch.equal(first, second)
    assert torch.equal(first[0, 2:], values[0, 2:])

