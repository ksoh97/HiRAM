from typing import Union, List
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                num_threads_torch: int = default_num_processes,
                                                                withSAM: bool = False, nnUNet_mask: Union[torch.Tensor, None] = None, 
                                                                SAM_mask: Union[torch.Tensor, None] = None):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
    
    if withSAM:
        predicted_probabilities = torch.sigmoid(torch.from_numpy(predicted_logits))
        segmentation = (predicted_probabilities > 0.5).squeeze(0).long()

        nnUNet_mask = configuration_manager.resampling_fn_probabilities(nnUNet_mask,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
        nnUNet_prob = label_manager.apply_inference_nonlin(nnUNet_mask)
        nnUNet_class = label_manager.convert_probabilities_to_segmentation(nnUNet_prob)
        nnUNet_class = nnUNet_class.cpu().numpy()

        nnUNet_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                            dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
        slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
        nnUNet_reverted_cropping[slicer] = nnUNet_class
        del nnUNet_class
        nnUNet_reverted_cropping = nnUNet_reverted_cropping.transpose(plans_manager.transpose_backward)

        SAM_reverted_cropping = None
        if SAM_mask is not None:
            SAM_mask = configuration_manager.resampling_fn_probabilities(SAM_mask,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
            SAM_prob = torch.sigmoid(torch.from_numpy(SAM_mask))
            SAM_class = (SAM_prob > 0.5).squeeze(0).long()
            SAM_class = SAM_class.cpu().numpy()

            SAM_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                                    dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
            SAM_reverted_cropping[slicer] = SAM_class
            del SAM_class
            SAM_reverted_cropping = SAM_reverted_cropping.transpose(plans_manager.transpose_backward)

    else:
        # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
        # apply_inference_nonlin will convert to torch
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        # del predicted_logits
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

        nnUNet_reverted_cropping = None
        SAM_reverted_cropping = None

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)

    torch.set_num_threads(old_threads)
    return segmentation_reverted_cropping, nnUNet_reverted_cropping, SAM_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  withSAM: bool = False, nnUNet_mask: Union[torch.Tensor, None] = None, 
                                  SAM_mask: Union[torch.Tensor, None] = None):
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        withSAM=withSAM, nnUNet_mask=nnUNet_mask, SAM_mask=SAM_mask
    )
    del predicted_array_or_file

    # save
    segmentation_final, nnUNet_mask_final, SAM_mask_final = ret
    del ret
    

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)
    if isinstance(nnUNet_mask_final, np.ndarray):
        rw.write_seg(nnUNet_mask_final, output_file_truncated + '_nnUNet_mask' + dataset_json_dict_or_file['file_ending'],
                     properties_dict)
    if SAM_mask_final is not None:
        rw.write_seg(SAM_mask_final, output_file_truncated + '_SAM_mask' + dataset_json_dict_or_file['file_ending'],
                     properties_dict)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
    torch.set_num_threads(old_threads)
