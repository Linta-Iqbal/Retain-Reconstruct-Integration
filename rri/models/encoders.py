"""Frozen pretrained encoders and MOSI sequence encoders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _transformers():
    try:
        from transformers import (  # type: ignore
            AutoImageProcessor,
            AutoModel,
            AutoTokenizer,
            CLIPModel,
            CLIPProcessor,
        )
    except ImportError as error:
        raise ImportError("Encoder use requires the 'transformers' package") from error
    return AutoImageProcessor, AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor


def _pool_model_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    raise TypeError(f"Unsupported encoder output: {type(output).__name__}")


class FrozenMultimodalEncoder(nn.Module):
    """CLIP or independent Hugging Face image/text encoders."""

    def __init__(
        self,
        *,
        mode: str = "clip",
        device: str | torch.device = "cpu",
        model_id: str = "openai/clip-vit-base-patch32",
        image_model_id: str | None = None,
        text_model_id: str | None = None,
        max_text_length: int = 77,
    ) -> None:
        super().__init__()
        AutoImageProcessor, AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor = _transformers()
        self.mode = mode
        self.device = torch.device(device)
        self.max_text_length = max_text_length
        if mode == "clip":
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.clip = CLIPModel.from_pretrained(model_id)
            self.image_processor = self.tokenizer = None
            self.image_model = self.text_model = None
        else:
            if not image_model_id or not text_model_id:
                raise ValueError("Independent mode requires image_model_id and text_model_id")
            self.processor = self.clip = None
            self.image_processor = AutoImageProcessor.from_pretrained(image_model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_id)
            self.image_model = AutoModel.from_pretrained(image_model_id)
            self.text_model = AutoModel.from_pretrained(text_model_id)
        self.to(self.device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def text_mask_token(self) -> str:
        if self.mode == "clip":
            return "[MASK]"
        return self.tokenizer.mask_token or "[MASK]"

    @torch.no_grad()
    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        if self.mode == "clip":
            inputs = self.processor(images=list(images), return_tensors="pt")
            features = self.clip.get_image_features(
                pixel_values=inputs["pixel_values"].to(self.device)
            )
        else:
            inputs = self.image_processor(images=list(images), return_tensors="pt")
            output = self.image_model(
                **{key: value.to(self.device) for key, value in inputs.items()}
            )
            features = _pool_model_output(output)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if self.mode == "clip":
            inputs = self.processor(
                text=list(texts), padding=True, truncation=True, return_tensors="pt"
            )
            features = self.clip.get_text_features(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )
        else:
            inputs = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            output = self.text_model(
                **{key: value.to(self.device) for key, value in inputs.items()}
            )
            features = _pool_model_output(output)
        return F.normalize(features.float(), dim=-1)


class FrozenVisionEncoder(nn.Module):
    """One frozen vision backbone shared by NYUD RGB and rendered depth."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        *,
        model_type: str = "clip",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        AutoImageProcessor, AutoModel, _, CLIPModel, CLIPProcessor = _transformers()
        self.model_type = model_type
        self.device = torch.device(device)
        if model_type == "clip":
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id)
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
        self.to(self.device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.model_type == "clip":
            features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            features = _pool_model_output(self.model(**inputs))
        return F.normalize(features.float(), dim=-1)


class BiGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if torch.any(lengths <= 0):
            raise ValueError("All sequence lengths must be positive")
        packed = pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=-1))


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        model_id: str,
        output_dim: int,
        *,
        max_length: int = 64,
        dropout: float = 0.2,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        _, AutoModel, AutoTokenizer, _, _ = _transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.backbone = AutoModel.from_pretrained(model_id)
        self.max_length = max_length
        self.freeze = freeze
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token or "[MASK]"

    def forward(self, word_lists: Sequence[Sequence[str]]) -> torch.Tensor:
        device = next(self.parameters()).device
        texts = [" ".join(words) for words in word_lists]
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                output = self.backbone(**tokens)
        else:
            output = self.backbone(**tokens)
        return self.projection(output.last_hidden_state[:, 0])

