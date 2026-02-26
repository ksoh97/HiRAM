# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
import math
from typing import Tuple, Type

from .common import MLPBlock_outdim
from .prompt_encoder import PositionEmbeddingRandom
from .HiRAM import HiRAM


class RAEblock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        sam_embedding_dim: int,
        cnn_embedding_dim: int,
        final_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.embedding_dim = sam_embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.attention_downsample_rate = attention_downsample_rate
        
        self.transformer = TwoWayAttentionBlock(
            embedding_dim=embedding_dim,
            sam_embedding_dim=sam_embedding_dim,
            cnn_embedding_dim=cnn_embedding_dim,
            num_heads=num_heads,
            final_dim=final_dim,
            mlp_dim=mlp_dim,
            activation=activation,
            skip_first_layer_pe=skip_first_layer_pe,
        )
        self.output_upscaling_sam = nn.Sequential(
            nn.ConvTranspose2d(sam_embedding_dim, final_dim, kernel_size=2, stride=2),
            activation(),
        )
        self.output_upscaling_cnn = nn.Sequential(
            nn.ConvTranspose2d(cnn_embedding_dim, final_dim, kernel_size=2, stride=2),
            activation(),
        )
        self.hiram = HiRAM(final_dim)
        
        self.pe_layer_sam = PositionEmbeddingRandom(sam_embedding_dim // 2)
        self.pe_layer_cnn = PositionEmbeddingRandom(cnn_embedding_dim // 2)
        self.sam_feat_proj = nn.Conv2d(256, sam_embedding_dim, kernel_size=1)
        
    def forward(
        self,
        sam_embedding: Tensor,
        cnn_embedding: Tensor,
        out_tokens: Tensor,
        tokens_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # Positional Embedding
        key_sam_pe = self.pe_layer_sam(sam_embedding.shape[-2:]).unsqueeze(0)
        key_cnn_pe = self.pe_layer_cnn(cnn_embedding.shape[-2:]).unsqueeze(0)
        
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        b, c_sam, h, w = sam_embedding.shape
        b, c_cnn, h, w = cnn_embedding.shape
        sam_embedding = sam_embedding.flatten(2).permute(0, 2, 1)
        cnn_embedding = cnn_embedding.flatten(2).permute(0, 2, 1)
        key_sam_pe = key_sam_pe.flatten(2).permute(0, 2, 1)
        key_cnn_pe = key_cnn_pe.flatten(2).permute(0, 2, 1)

        # Apply transformer blocks
        queries, keys_sam, keys_cnn = self.transformer(
            queries=out_tokens, keys_sam=sam_embedding, keys_cnn=cnn_embedding, query_pe=tokens_pe, 
            key_sam_pe=key_sam_pe, key_cnn_pe=key_cnn_pe,)

        # Run HiRAM
        keys_sam = self.output_upscaling_sam(keys_sam.permute(0, 2, 1).reshape(b, c_sam, h, w))
        keys_cnn = self.output_upscaling_cnn(keys_cnn.permute(0, 2, 1).reshape(b, c_cnn, h, w))
        keys_sam, keys_cnn, region_masks = self.hiram(keys_sam, keys_cnn)

        return queries, keys_sam, keys_cnn, region_masks


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        sam_embedding_dim: int,
        cnn_embedding_dim: int,
        num_heads: int,
        final_dim: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim * 2, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim * 2)

        self.cross_attn_token_to_image_sam = Attention_dimchange(embedding_dim, sam_embedding_dim, embedding_dim//2, num_heads)
        self.cross_attn_token_to_image_cnn = Attention_dimchange(embedding_dim, cnn_embedding_dim, embedding_dim//2, num_heads)
        self.norm2_sam = nn.LayerNorm(embedding_dim)
        self.norm2_cnn = nn.LayerNorm(embedding_dim)

        self.mlp_sam = MLPBlock_outdim(embedding_dim, mlp_dim, embedding_dim, activation)
        self.mlp_cnn = MLPBlock_outdim(embedding_dim, mlp_dim, embedding_dim, activation)
        self.norm3_sam = nn.LayerNorm(embedding_dim)
        self.norm3_cnn = nn.LayerNorm(embedding_dim)
          
        self.norm4_sam = nn.LayerNorm(sam_embedding_dim)
        self.norm4_cnn = nn.LayerNorm(cnn_embedding_dim)
        self.cross_attn_image_to_token_sam = Attention_dimchange(sam_embedding_dim, embedding_dim, embedding_dim//2, num_heads)
        self.cross_attn_image_to_token_cnn = Attention_dimchange(cnn_embedding_dim, embedding_dim, embedding_dim//2, num_heads)
        
        self.linear = nn.Linear(embedding_dim*2, final_dim*2)
        self.skip_first_layer_pe = skip_first_layer_pe
        
    def forward(
        self, queries: Tensor, keys_sam: Tensor, keys_cnn: Tensor, query_pe: Tensor, key_sam_pe: Tensor, key_cnn_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
                
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        
        # Output tokens Speration
        queries_sam = queries[:, :, :queries.shape[-1]//2]
        queries_cnn = queries[:, :, queries.shape[-1]//2:]
        query_pe_sam = query_pe[:, :, :queries.shape[-1]//2]
        query_pe_cnn = query_pe[:, :, queries.shape[-1]//2:]
        
        # Cross attention block, tokens attending to SAM image embedding
        q_sam = queries_sam + query_pe_sam
        k_sam = keys_sam + key_sam_pe
        attn_out_sam = self.cross_attn_token_to_image_sam(q=q_sam, k=k_sam, v=keys_sam)
        queries_sam = queries_sam + attn_out_sam
        queries_sam = self.norm2_sam(queries_sam)
        
        # Cross attention block, tokens attending to CNN image embedding
        q_cnn = queries_cnn + query_pe_cnn
        k_cnn = keys_cnn + key_cnn_pe
        attn_out_cnn = self.cross_attn_token_to_image_cnn(q=q_cnn, k=k_cnn, v=keys_cnn)
        queries_cnn = queries_cnn + attn_out_cnn
        queries_cnn = self.norm2_cnn(queries_cnn)

        # MLP block for SAM
        mlp_out_sam = self.mlp_sam(queries_sam)
        queries_sam = queries_sam + mlp_out_sam
        queries_sam = self.norm3_sam(queries_sam)
        
        # MLP block for CNN
        mlp_out_cnn = self.mlp_cnn(queries_cnn)
        queries_cnn = queries_cnn + mlp_out_cnn
        queries_cnn = self.norm3_cnn(queries_cnn)
        
        # Output tokens Aggregation
        queries = torch.cat([queries_sam, queries_cnn], dim=2)
        
        # Cross attention block, SAM image embedding attending to tokens
        q_sam = queries_sam + query_pe_sam
        k_sam = keys_sam + key_sam_pe
        attn_out_sam = self.cross_attn_image_to_token_sam(q=k_sam, k=q_sam, v=queries_sam)
        keys_sam = keys_sam + attn_out_sam
        keys_sam = self.norm4_sam(keys_sam)
        
        # Cross attention block, CNN image embedding attending to tokens
        q_cnn = queries_cnn + query_pe_cnn
        k_cnn = keys_cnn + key_cnn_pe
        attn_out_cnn = self.cross_attn_image_to_token_cnn(q=k_cnn, k=q_cnn, v=queries_cnn)
        keys_cnn = keys_cnn + attn_out_cnn
        keys_cnn = self.norm4_cnn(keys_cnn)
        
        return queries, keys_sam, keys_cnn



class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class Attention_dimchange(nn.Module):
    def __init__(
        self,
        q_embedding_dim: int,
        k_embedding_dim: int,
        internal_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = q_embedding_dim
        self.internal_dim = internal_dim
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(q_embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(k_embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(k_embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, q_embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, query_mask: Tensor = None, key_mask: Tensor = None) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)

        if key_mask is not None:
            log_k = torch.log(key_mask.clamp(min=1e-8))
            attn = attn + log_k[:, None, None, :]
        
        if query_mask is not None:
            log_q = torch.log(query_mask.clamp(min=1e-8))
            attn = attn + log_q[:, None, :, None]

        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out    

