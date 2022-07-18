This is the code release for the paper All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines.

**1. Environment:**

This code is tested under following environment

- NVIDIA A100 GPU (40GB memory) Clusters 

- Pytorch 1.9.0; Torchvision 0.10.0
- CUDA11.1 toolkit

To install the environment run the following pip command:

- pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html




**2. Training of mapping network from Raw to RGB**

We provide two ways for training the RAW2RGB mapping network, depending on the computational resource you have access to 

- For best performance, clone the Pynet training workflow https://github.com/aiff22/PyNET-PyTorch and follow the instruction to perform coarse to fine training. Note that you need to replace the model definition file (model.py) with our model.py file as we slightly modified the model structure. 

- The coarse to fine training, however, is computationally heavy. We also provide a fast training flow (see train_decoder_fast method in model class). Run the following command to conduct a fast training.

- python3 main.py --command train_decoder --fast_training True --visual False --mode train

**3. Training of mapping network from RGB to RAW**

Run the following command to train the mapping network from RGB to Raw

- python3 main.py --command train_encoder --train_encoder True --mode train

**4. Pre-trained weight**

If you want to use pre-trained weight, download them from https://drive.google.com/drive/folders/1G9ElfOO_pKLTP_EWK8Unsmg-Pzzvpaoy?usp=sharing and put it in ckpt/ folder. 

**5. Inference** 

Run to the following command to perform inference. Note you need to download Imagenet and specify the local path of data before running the script

- python3 main.py --command eval.
