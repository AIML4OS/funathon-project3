"""
SegFormer model wrappers.

- `SemanticSegmentationSegformer` is a thin subclass of HuggingFace's
  `SegformerPreTrainedModel` that exposes a cleaner `forward()` and
  freeze/unfreeze helpers for transfer learning.
- `SegformerB5` is a factory: it fetches the CLC+ label mapping from S3,
  patches the `nvidia/mit-b5` config for 14-channel input (12 L2A spectral bands + NDVI + NDWI) and 10 land-cover
  classes, then loads the pretrained ImageNet weights.
"""

from typing import Optional

import requests
import torch
from torch import nn
from transformers import (
    SegformerConfig,
    SegformerDecodeHead,
    SegformerModel,
    SegformerPreTrainedModel,
)


class SemanticSegmentationSegformer(SegformerPreTrainedModel):
    def __init__(self, config, logits: bool = True):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.logits = logits

        # Initialize weights and apply final processing
        self.post_init()

    def freeze(self):
        """
        Freeze encoder parameters.
        """
        for param in self.segformer.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze encoder parameters.
        """
        for param in self.segformer.parameters():
            param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward method.
        """
        outputs = self.segformer(
            pixel_values,
            output_attentions=False,
            # All four encoder feature maps are fed to the all-MLP decoder, which
            # fuses local detail (stage 1, H/4) with global context (stage 4, H/32).
            output_hidden_states=True,
            return_dict=True,
        )
        encoder_hidden_states = outputs.hidden_states
        logits = self.decode_head(encoder_hidden_states)

        # When labels are supplied (training), upsample logits back to the label
        # resolution so cross-entropy can be computed pixel-for-pixel. At
        # inference (labels=None) we return the raw H/4 logits — the caller
        # decides whether to upsample (see src.inference.prediction make_prediction).
        if labels is not None:
            return nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            return logits


class SegformerB5(SemanticSegmentationSegformer):
    """
    SegformerB5 model.
    """

    def __new__(
        cls,
        n_bands="14",
        logits: bool = True,
        freeze_encoder: bool = False,
    ):
        id2label = requests.get(
            "https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/clcplus-backbone-id2label.json"
        ).json()
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

        config = SegformerConfig.from_pretrained("nvidia/mit-b5")
        config.num_channels = int(n_bands)  # Pre-baked GeoTIFFs: 14 channels (12 L2A spectral + NDVI + NDWI), vs 3 for RGB
        config.num_labels = len(id2label)
        config.id2label = id2label
        config.label2id = label2id

        # ignore_mismatched_sizes=True lets us load the pretrained MiT-B5 weights
        # even though the very first patch-embedding layer has a different shape
        # in our config (14 input channels vs 3 in the ImageNet pretraining). That
        # one layer is reinitialised from scratch; all the deeper transformer
        # weights are kept — which is the whole point of transfer learning here.
        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b5",
            config=config,
            ignore_mismatched_sizes=True,
        )
        if freeze_encoder:
            model.freeze()
        return model
