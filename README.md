# Multi-scale Synergism Ensemble Progressive and Contrastive Investigation for Image Restoration

[Zhiying Jiang](https://scholar.google.com/citations?user=uK6WHa0AAAAJ&hl=zh-CN&oi=ao)&#8224;, [Shuzhou Yang](https://ysz2022.github.io/)&#8224;, [Jinyuan Liu](https://scholar.google.com/citations?user=a1xipwYAAAAJ&hl=zh-CN&oi=ao), [Xin Fan](https://scholar.google.com/citations?user=vLN1njoAAAAJ&hl=zh-CN), [Risheng Liu](https://rsliu.tech/)* (&#8224;equal contribution, *corresponding author)



## Prerequisites
- Linux or macOS
- Python 3.7
- NVIDIA GPU + CUDA CuDNN

## 🔑 Setup
Type the command:
```
pip install -r requirements.txt
```

## 🧩 Download
You need **create** a directory `./logs/[YOUR-MODEL]` (e.g., `./logs/SEPC_derainL`). \
Download the pre-trained model and put it into `./logs/[YOUR-MODEL]`. \
Here we release the pre-trained model trained on [Rain100L](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html):
- [**Model**](https://drive.google.com/file/d/1bbEHVtVew6JCnwgktXJurJxibaPkQlNG/view?usp=sharing)

## 🚀 Quick Run
- You need **create** a directory `./testData` and put the degraded images to it.
- Test the model with the pre-trained weights:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```
- The test results will be saved to a directory here: `./results`.
