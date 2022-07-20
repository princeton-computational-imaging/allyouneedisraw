# All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines

<img src="https://github.com/princeton-computational-imaging/allyouneedisraw/blob/main/img/defense.png" width="450">

This is the code release for the paper [All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines](https://arxiv.org/pdf/2112.09219.pdf). 

## Summary
Existing neural networks for computer vision tasks are vulnerable to adversarial attacks: adding imperceptible perturbations to the input images can fool these models into making a false prediction on an image that was correctly predicted without the perturbation. Various defense methods have proposed image-to-image mapping methods, either including these perturbations in the training process or removing them in a preprocessing step. In doing so, existing methods often ignore that the natural RGB images in today's datasets are not captured but, in fact, recovered from RAW color filter array captures that are subject to various degradations in the capture. In this work, we exploit this RAW data distribution as an empirical prior for adversarial defense. Specifically, we proposed a model-agnostic adversarial defensive method, which maps the input RGB images to Bayer RAW space and back to output RGB using a learned camera image signal processing (ISP) pipeline to eliminate potential adversarial patterns. The proposed method acts as an off-the-shelf preprocessing module and, unlike model-specific adversarial training methods, does not require adversarial images to train. As a result, the method generalizes to unseen tasks without additional retraining. Experiments on large-scale datasets (e.g., ImageNet, COCO) for different vision tasks (e.g., classification, semantic segmentation, object detection) validate that the method significantly outperforms existing methods across task domains.

## Getting started
To get started, first clone this repository in your local directory using 

```
https://github.com/princeton-computational-imaging/allyouneedisraw
```

Then install the necessary environment, specifically, this code is tested under following environment

- NVIDIA A100 GPU (40GB memory) Clusters 
- Pytorch 1.9.0; Torchvision 0.10.0
- CUDA11.1 toolkit

To install the environment run the following pip command:

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Training
**1. Training of mapping network from Raw to RGB**

We provide two ways for training the RAW2RGB mapping network, depending on the computational resource you have access to 

- For best performance, clone the Pynet training workflow https://github.com/aiff22/PyNET-PyTorch and follow the instruction to perform coarse to fine training. Note that you need to replace the model definition file (model.py) with our model.py file as we slightly modified the model structure. 

- The coarse to fine training, however, is computationally heavy. We also provide a fast training flow (see train_decoder_fast method in model class). Run the following command to conduct a fast training.

```
python3 main.py --command train_decoder --fast_training True --visual False --mode train

```

**2. Training of mapping network from RGB to RAW**

Run the following command to train the mapping network from RGB to Raw

```
python3 main.py --command train_encoder --train_encoder True --mode train

```
## Evaluation

Run to the following command to perform inference. Note you need to download Imagenet and specify the local path of data before running the script. If you have not trained the model by yourself, please download the pre-trained weight before running the script. 

```
python3 main.py --command eval

```

## Pre-trained weight

If you want to use pre-trained weight, download them from https://drive.google.com/drive/folders/1G9ElfOO_pKLTP_EWK8Unsmg-Pzzvpaoy?usp=sharing and put it in ckpt/ folder. 

## Reference
If you find our work/code useful, please consider citing our paper:

```bib
@inproceedings{zhang2022raw,
  title={All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines},
  author={Zhang, Yuxuan and Dong, Bo and Heide, Felix},
  booktitle={European conference on computer vision},
  year={2022}
}
```

## Acknowledgements

This code in parts is borrowed from [PyNET](https://github.com/aiff22/PyNET-PyTorch).

