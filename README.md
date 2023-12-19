<div align="center">
  
# 【TIM'2023🔥】Multi-scale Synergism Ensemble Progressive and Contrastive Investigation for Image Restoration
[![Journal](http://img.shields.io/badge/TIM-2023-FFD93D.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=19)
[![Paper](http://img.shields.io/badge/Paper-IEEE-FF6B6B.svg)](https://ieeexplore.ieee.org/document/10363208)
</div>

Welcome! This is the official implementation of our paper: [**Multi-scale Synergism Ensemble Progressive and Contrastive Investigation for Image Restoration**](https://ieeexplore.ieee.org/document/10363208)

Authors: [Zhiying Jiang](https://scholar.google.com/citations?user=uK6WHa0AAAAJ&hl=zh-CN&oi=ao)&#8224;, [Shuzhou Yang](https://ysz2022.github.io/)&#8224;, [Jinyuan Liu](https://scholar.google.com/citations?user=a1xipwYAAAAJ&hl=zh-CN&oi=ao), [Xin Fan](https://scholar.google.com/citations?user=vLN1njoAAAAJ&hl=zh-CN), [Risheng Liu](https://rsliu.tech/)* (&#8224;equal contribution, *corresponding author)



## Prerequisites
- Linux or macOS
- Python 3.7
- NVIDIA GPU + CUDA CuDNN

## 🔑 Setup
Type the command:
```
conda create -n SEPC python=3.7
conda activate SEPC
pip install -r requirements.txt
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

## 🧩 Download
You need **create** a directory `./logs/[YOUR-MODEL]` (e.g., `./logs/SEPC_derainL`). \
Download the pre-trained model and put it into `./logs/[YOUR-MODEL]`. \
Here we release the pre-trained model trained on [Rain100L](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) and [Rain100H](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html):
- [**Model(Rain100L)**](https://drive.google.com/file/d/1bbEHVtVew6JCnwgktXJurJxibaPkQlNG/view?usp=sharing)
- [**Model(Rain100H)**](https://drive.google.com/file/d/1ZpdjnK-YLtYJZsPHfZU6FxxVbklJtgTn/view?usp=sharing)

## 🚀 Quick Run
- You need **create** a directory `./testData` and put the degraded images to it.
- Test the model with the pre-trained weights:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```
- The test results will be saved to a directory here: `./results`.

## 🤖 Training
- You need **create** a directory `./trainData` and put the degraded training data to it.
- Train a model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```
- Loss curve and checkpoint can be found in the directory `./log`.
