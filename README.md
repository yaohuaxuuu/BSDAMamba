# BSDAMamba
Enhancing Medical Image Classification with BSDA-Mamba: Integrating Bayesian Random Semantic Data Augmentation and Residual Connections

# Summary
In this study, we introduce BSDA-Mamba, a novel medical image classification approach that integrates Bayesian Random Semantic Data Augmentation (BSDA) with the MedMamba model, enhanced by residual connection blocks. BSDA augments medical image data semantically, enhancing the model's generalization ability and classification performance. MedMamba, a deep learning-based state space model, excels in capturing long-range dependencies in medical images. By incorporating residual connections, BSDA-Mamba further improves feature extraction capabilities. Through comprehensive experiments on eight medical image datasets, we demonstrate that BSDA-Mamba outperforms existing models in terms of accuracy, area under the curve, and F1-score. Our results highlight BSDA-Mamba's potential as a reliable tool for medical image analysis, particularly in handling diverse imaging modalities from X-rays to MRI.

#Run
python train.py

#Related technologies
[Vmamba](https://github.com/MzeroMiko/VMamba))
[MedMamba](https://github.com/YubiaoYue/MedMamba)
[BSDA](https://github.com/YaoyaoZhu19/BSDA)

# The classification performance of BSDAMamba
| Dataset | Task | Accuracy | AUC | F1-score |
|:------:|:--------:|:--------:|:----------:|:----------:|
| ***Blood*** | Multi-Class|98.1|99.9|98.0|
| ***Brain*** | Multi-Class|96.6|99.5|95.8|
| ***DermaMNIST*** | Multi-Class|79.9|93.2|78.5|
| ***PneumoniaMNIST*** |Multi-Class|95.6|98.7|95.6|
| ***PathMNIST*** |Multi-Class|89.7|88.5|88.9|
| ***OCTMNIST*** |Multi-Class|98.3|99.9|99.8|
| ***TissueMNIST*** |Multi-Class|92.8|99.2|93.0|

#The algorithm and Implementation of the BSDA-Mamba.

![image](https://github.com/user-attachments/assets/9945c330-3b5a-434b-8704-257b062d83cb)

#Datasets

![image](https://github.com/user-attachments/assets/1e70e571-e849-40f9-8662-e681f6096619)
![Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
![Brain-Tumor-BRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
![MedMNIST](https://github.com/MedMNIST/MedMNIST)

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
