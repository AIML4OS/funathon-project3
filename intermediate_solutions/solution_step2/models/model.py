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
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )
        encoder_hidden_states = outputs.hidden_states
        logits = self.decode_head(encoder_hidden_states)

        if labels is not None:
            # upsample logits to the images' original size
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
        type_labeler: str = "CLCplus-Backbone",
    ):
        id2label = requests.get(
            f"https://minio.lab.sspcloud.fr/projet-hackathon-ntts-2025/data-label/{type_labeler}/{type_labeler.lower()}-id2label.json"
        ).json()
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

        config = SegformerConfig.from_pretrained("nvidia/mit-b5")
        config.num_channels = int(n_bands)  # Update number of input channels
        config.num_labels = len(id2label)  # Set number of segmentation classes
        config.id2label = id2label  # Assign id2label mapping
        config.label2id = label2id  # Assign label2id mapping

        model = SemanticSegmentationSegformer.from_pretrained(
            "nvidia/mit-b5",
            config=config,
            ignore_mismatched_sizes=True,  # Prevent errors due to num_channels change
        )
        if freeze_encoder:
            model.freeze()
        return model
