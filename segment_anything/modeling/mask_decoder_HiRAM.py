# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Type

from .common import LayerNorm2d, MLPBlock_outdim
from .RAEblock import RAEblock


class MaskDecoder_HiRAM(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        sam_embedding_dim: list,
        cnn_embedding_dim: list,
        transformer_args: dict,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = embedding_dim
        self.sam_embedding_dim = sam_embedding_dim
        self.cnn_embedding_dim = cnn_embedding_dim
        self.mlp_dim = transformer_args.get("mlp_dim", 2048)
        self.num_heads = transformer_args.get("num_heads", 8)
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, embedding_dim * 2)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, embedding_dim * 2)
        
        self.cnn_fusion = nn.Sequential(
            nn.Conv2d(992, cnn_embedding_dim[0] * 2, kernel_size=3, padding=1),
            LayerNorm2d(cnn_embedding_dim[0] * 2),
            activation(),
            nn.Conv2d(cnn_embedding_dim[0] * 2, cnn_embedding_dim[0], kernel_size=3, padding=1),
            LayerNorm2d(cnn_embedding_dim[0]),
        )
        
        self.stage_1 = RAEblock(
            embedding_dim=embedding_dim,
            sam_embedding_dim=sam_embedding_dim[0],
            cnn_embedding_dim=cnn_embedding_dim[0],
            final_dim=sam_embedding_dim[1],
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            activation=activation,
            skip_first_layer_pe=True,
        )
        self.stage_2 = RAEblock(
            embedding_dim=embedding_dim,
            sam_embedding_dim=sam_embedding_dim[1],
            cnn_embedding_dim=cnn_embedding_dim[1],
            final_dim=sam_embedding_dim[2],
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            activation=activation,
            skip_first_layer_pe=False,
        )
        self.tokens_mlp = MLPBlock_outdim(embedding_dim*2, self.mlp_dim, sam_embedding_dim[2]*2, activation)
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(embedding_dim, embedding_dim, embedding_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        cnn_embeddings: list,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, region_masks = self.predict_masks(
            image_embeddings=image_embeddings,
            cnn_embeddings=cnn_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        
        # Prepare output
        return masks, region_masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        cnn_embeddings: list,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = output_tokens

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        src = src + dense_prompt_embeddings
        
        # Run the Decoder block
        tokens_pe_1 = tokens_pe_2 = tokens
        cnn_embeddings = self.cnn_fusion(cnn_embeddings)
        tokens_1, sam_1, cnn_1, region_masks_1 = self.stage_1(image_embeddings, cnn_embeddings, tokens, tokens_pe_1)
        tokens_2, sam_2, cnn_2, region_masks_2 = self.stage_2(sam_1, cnn_1, tokens_1, tokens_pe_2)
        
        hs = self.tokens_mlp(tokens_2[:, 0:1, :])
        hs_sam = hs[:, :, :hs.shape[-1]//2]
        hs_cnn = hs[:, :, hs.shape[-1]//2:]
        masks_sam = (hs_sam @ sam_2.view(b, -1, 256 * 256)).view(b, -1, 256, 256)
        masks_cnn = (hs_cnn @ cnn_2.view(b, -1, 256 * 256)).view(b, -1, 256, 256)
        masks = masks_sam*0.5 + masks_cnn*0.5

        return masks, [region_masks_1, region_masks_2]




# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py #
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
