import pydoc
import torch
import numpy as np
import torch.nn.functional as F
from typing import Union, Type, List, Tuple
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from segment_anything import sam_model_registry
from segment_anything.utils.LoRA import LoRA_Sam
from segment_anything.utils.transforms import ResizeLongestSide


def get_network_from_plans_withSAM(arch_kwargs, arch_kwargs_req_import, input_channels, output_channels, 
                                   SAM_config, allow_init=True, deep_supervision: Union[bool, None] = None,):
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision

    network = PlainConvUNet_withSAM(
        SAM_config=SAM_config,
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


class Processor:
    def __init__(self, model_input_size):
        super().__init__
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = ResizeLongestSide(model_input_size)
        self.target_length = model_input_size
        self.reset_image()
    
    def __call__(self, image, label, prompt: list) -> dict:
        """
        Return: 
            inputs = {
                "image": image_torch, 
                "original_size": self.original_size,
                "point_coords": prompt_torch["Point"],
                "point_labels": prompt_torch["Point_label"],
                "boxes": prompt_torch["Box"],
                "mask_inputs": prompt_torch["Mask"],
                "origin_prompt" : prompt
            }
        """
        image_torch, label_torch = self.process_image(image, label)
        prompt_torch = self.process_prompt(prompt)

        inputs = {"image": image_torch, 
                  "nnUNet_mask": label_torch,
                  "original_size": self.original_size,
                  "origin_prompt" : prompt}
        
        if prompt_torch.get("Point") is not None:
            inputs["point_coords"] = prompt_torch["Point"]
            inputs["point_labels"] = prompt_torch["Point_label"]
        
        if prompt_torch.get("Box") is not None:
            inputs["boxes"] = prompt_torch["Box"]
        
        if prompt_torch.get("Mask") is not None:
            inputs["mask_inputs"] = prompt_torch["Mask"]
        
        return inputs
    
    def process_image(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the image to make it to the input format of SAM
        """
        input_image_torch = self.transform_apply_image(image)
        input_label_torch = self.transform_apply_image(label)

        self.original_size = image.shape[-2:] #  (H,W)
        self.input_size = input_image_torch.shape[-2:]
        return input_image_torch, input_label_torch

    def process_prompt(self, prompt) -> torch.tensor:
        point, box, mask = prompt['Point'], prompt['Box'], prompt['Mask']
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

        if point is not None:
            point_coords = np.array(point)
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(prompt['Point_label'], dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
        
        if box is not None:
            box = np.array(box)
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[:, None, :]
        
        if mask is not None:
            # mask = F.interpolate(mask.unsqueeze(1), size=(256, 256), mode='nearest').to(torch.float16)
            mask = F.interpolate(mask.unsqueeze(1), size=(256, 256), mode='nearest').to(torch.float)
            mask_input_torch = mask.squeeze(0)

        prompt_torch = {
            'Point': coords_torch,
            'Point_label': labels_torch,
            'Box': box_torch,
            'Mask': mask_input_torch,
        }
        return prompt_torch
    
    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    def transform_apply_image(self, image: torch.Tensor) -> torch.Tensor:
        C, H, W = image.shape
        scale = self.target_length / max(H, W)
        new_h, new_w = int(round(H * scale + 0.5)), int(round(W * scale + 0.5))

        image = image.unsqueeze(0).float()
        resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return resized.squeeze(0)


class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thres):
        return (input > thres).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PlainConvUNet_withSAM(nn.Module):
    def __init__(self,
                 SAM_config: dict,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        self.SAM_config = SAM_config

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)
    
        if SAM_config:
            self.SAM = sam_model_registry[SAM_config['model_type']]()

            for module_name in ['image_encoder', 'prompt_encoder', 'mask_decoder']:
                update_flag = SAM_config["Update"].get(module_name, False)    
                module = getattr(self.SAM, module_name)
                for param in module.parameters():
                    param.requires_grad = update_flag
                        
            if SAM_config['model_name'] == 'LoRA':
                self.SAM = LoRA_Sam(self.SAM, r=4)
                self.processor = Processor(self.SAM.sam.image_encoder.img_size)
            else:    
                self.processor = Processor(self.SAM.image_encoder.img_size)

    def forward(self, x):
        ## Run nnUNet
        nnUNet_skips = self.encoder(x)
        nnUNet_outs = self.decoder(nnUNet_skips)

        cnn_feats = []
        for cnn_feat in nnUNet_skips[:-1]:
            cnn_feat = F.interpolate(cnn_feat, size=(64,64), mode="bilinear", align_corners=False)
            cnn_feats.append(cnn_feat)
        cnn_feats = torch.concat(cnn_feats, dim=1)
        
        if self.decoder.deep_supervision:
            nnUNet_out = nnUNet_outs[0]
        else:
            nnUNet_out = nnUNet_outs
        probs = torch.softmax(nnUNet_out, dim=1)
        nnUNet_mask = BinarizeSTE.apply(probs[:, 1:2, :, :], 0.5)

        SAM_inputs = []
        for i in range(x.shape[0]):
            data_slice = ((x[i] - x[i].min()) / (x[i].max() - x[i].min())) * 255
            data_slice = data_slice.to(torch.uint8).repeat(3, 1, 1)
            label = nnUNet_mask[i]
            prompts = {
                'Point': None,
                'Point_label': None,
                'Box': None,
                'Mask': label,
            }
            processor_input = self.processor(data_slice, label, prompts)
            processor_input["cnn_feats"] = cnn_feats[i].unsqueeze(0)
            SAM_inputs.append(processor_input)

        ## Run SAM
        SAM_output = self.SAM(SAM_inputs, multimask_output=False)
        SAM_outs = torch.stack([out["low_res_logits"].sum(dim=0, keepdim=True) if out["low_res_logits"].shape[0] > 1 else out["low_res_logits"] 
                                for out in SAM_output], dim=0).squeeze(1)
        region_masks = torch.stack([out["low_regions_logprobs"] for out in SAM_output], dim=0)
        
        return nnUNet_outs, nnUNet_mask, SAM_outs, region_masks

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

