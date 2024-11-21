# BSDAMamba
Enhancing Medical Image Classification with BSDA-Mamba: Integrating Bayesian Random Semantic Data Augmentation and Residual Connections

# Summary
In this study, we introduce BSDA-Mamba, a novel medical image classification approach that integrates Bayesian Random Semantic Data Augmentation (BSDA) with the MedMamba model, enhanced by residual connection blocks. BSDA augments medical image data semantically, enhancing the model's generalization ability and classification performance. MedMamba, a deep learning-based state space model, excels in capturing long-range dependencies in medical images. By incorporating residual connections, BSDA-Mamba further improves feature extraction capabilities. Through comprehensive experiments on eight medical image datasets, we demonstrate that BSDA-Mamba outperforms existing models in terms of accuracy, area under the curve, and F1-score. Our results highlight BSDA-Mamba's potential as a reliable tool for medical image analysis, particularly in handling diverse imaging modalities from X-rays to MRI.

#Related technologies
[Vmamba](https://github.com/MzeroMiko/VMamba))
[MedMamba](https://github.com/YubiaoYue/MedMamba)
[BSDA](https://github.com/YaoyaoZhu19/BSDA)

# Installation
* `pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`
* `pip install packaging`
* `pip install timm==0.4.12`
* `pip install pytest chardet yacs termcolor`
* `pip install submitit tensorboardX`
* `pip install triton==2.0.0`
* `pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs`
## Other requirements:
* Linux System
* NVIDIA GPU
* CUDA 12.0+
