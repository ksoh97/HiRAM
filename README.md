# HiRAM
Repository for source code submission (Anonymous Version)

## Abstract
Accurate white matter (WM) lesion segmentation is essential for diagnosing and monitoring neurological disorders such as white matter hyperintensities and multiple sclerosis. However, this task remains highly challenging owing to the small and spatially scattered nature of lesions and their low contrast with surrounding tissues. Recently, the segment anything model (SAM) and its medical variants, often augmented with CNNs or Mamba, have shown promising potential for medical image segmentation, yet they struggle to reliably capture subtle lesion patterns and indistinct boundaries that demand fine-grained regional discrimination. In this work, we propose a novel hierarchical region-aware multi-granularity Mamba (HiRAM) incorporated into the SAM decoder. HiRAM explicitly decomposes the features into region-specific representations (i.e., interior, boundary, and background regions) and performs confidence-driven, non-causal state-space propagation. By incorporating region-aware modeling that prioritizes semantically reliable features and hierarchically integrates multi-granularity learning, our method enhances boundary sensitivity and fine-grained lesion delineation while preserving global contextual consistency. Comprehensive evaluations across two public challenge datasets reveal that HiRAM outperforms state-of-the-art methods, achieving unmatched segmentation accuracy and robustness.

## Key Components
- HiRAM_config.yaml: configuration file
- HiRAM_run.py: training entrypoint
- nnunetv2/inference/predict_from_raw_data.py: inference entrypoint
- nnunetv2/utilities/get_network_from_plans.py: overall structure
- segment_anything/modeling/: proposed modules and SAM backbone
