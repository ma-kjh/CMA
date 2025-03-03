# [CVPR2025]Enhanced OoD Detection through Cross-Modal Alignment of Multimodal Representations

This repository is the official implementation of [CMA](https://github.com/ma-kjh/CMA/tree/main)

> **Enhanced OoD Detection through Cross-Modal Alignment of Multimodal Representations** \
> [Jeonghyeon Kim](https://scholar.google.co.kr/citations?user=u6DjYLsAAAAJ&hl=ko), [Sangheum Hwang](https://scholar.google.co.kr/citations?user=QtI8XmgAAAAJ&hl=ko)

\[[arXiv Paper]()\]

<p align="center">
     <img alt="main" src="https://github.com/ma-kjh/CMA/blob/main/main.jpg" width="50%"/>
</p>


Abstract: _Prior research on out-of-distribution detection (OoDD) has primarily focused on single-modality models. Recently, with the advent of large-scale pretrained vision-language models such as CLIP, OoDD methods utilizing such multi-modal representations through zero-shot and prompt learning strategies have emerged. However, these methods typically involve either freezing the pretrained weights or only partially tuning them, which can be suboptimal for downstream datasets. In this paper, we highlight that multi-modal fine-tuning (MMFT) can achieve notable OoDD performance. Despite some recent works demonstrating the impact of fine-tuning methods for OoDD, there remains significant potential for performance improvement. We investigate the limitation of na\"ive fine-tuning methods, examining why they fail to fully leverage the pretrained knowledge. Our empirical analysis suggests that this issue could stem from the modality gap within in-distribution (ID) embeddings. To address this, we propose a training objective that enhances cross-modal alignment by regularizing the distances between image and text embeddings of ID data. This adjustment helps in better utilizing pretrained textual information by aligning similar semantics from different modalities (i.e., text and image) more closely in the hyperspherical representation space. We theoretically demonstrate that the proposed regularization corresponds to the maximum likelihood estimation of an energy-based model on a hypersphere. Utilizing ImageNet-1k OoD benchmark datasets, we show that our method, combined with post-hoc OoDD approaches leveraging pretrained knowledge (e.g., NegLabel), significantly outperforms existing methods, achieving state-of-the-art OoDD performance and leading ID accuracy._

## Requirements
- python == 3.9.18
- torch == 1.12.0+cu116
- torchvision == 0.13.0+cu116
- numpy == 1.25.2
- scikit-learn == 1.4.2

## Dataset Preparation
The guidelines are provided in [MOS](https://github.com/deeplearning-wisc/large_scale_ood?tab=readme-ov-file) and [OpenOOD](https://github.com/Jingkang50/OpenOOD) GitHub repositories to download the datasets directly for our experiments.

### In-distribution datasets
For fine-tuning and evaluation [ImageNet-1k](https://image-net.org/challenges/LSVRC/2012/index) setting, please prepare datasets in `<data_dir>` as follows:

```
|---- data_dir/
|      |---- imagenet_1k/
|            |---- train/
|                  |---- ...
|            |---- val/
|                  |---- ...
```

### Out-of-distribution Datasets
Following MOS and OpenOOD, we use the following OoD datasets:

**MOS:**
- iNaturalist
- SUN
- Places
- Textures

**OpenOOD v1.5:**
- SSB-hard
- NINCO
- iNaturalist
- Textures
- OpenImage-O


## NegLabels
We utilize negative labels from the WordNet database, located in `./NegLabel/txtfiles`.

Our approach employs the same negmining method as described in [NegLabel](https://github.com/XueJiang16/NegLabel/tree/main). 

The extracted 10,000 texts are located in `./NegLabel/neg_text_10000.npy` (Our default prompt is "The nice")
     
## Multi-modal Fine-tuning
DDP code will be released soon. Please stay tuned!

### [FLYP: Finetune Like You Pretrain](https://github.com/locuslab/FLYP) 

```python
image_embeddings, text_embeddings, scale = model(images, texts)
norm_image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
norm_text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

logits_per_image = (scale) * (norm_image_embeddings @ norm_text_embeddings.T)
logits_per_text = logits_per_image.T

# ground_truth_text = torch.arange(# of batch size of image, text pairs)
# loss_img = torch.nn.CrossEntropyLoss()
# loss_txt = torch.nn.CrossEntropyLoss()
image_loss = loss_img(logits_per_image, ground_truth_text)
text_loss = loss_txt(logits_per_text, ground_truth_text)
            
total_loss = (image_loss + text_loss )/2  
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --is-train --prompt-name <prompt> --lam1 0.0 --epochs 10 --bs 512 --lr 1e-5
```

### Cross-Modal-Alignment (CMA)
```python
CMA_text = -torch.logsumexp(logits_per_text,dim=1)
CMA_image = -torch.logsumexp(logits_per_image,dim=1)

total_loss = (image_loss + args.lam * CMA_image.mean())/2 + (text_loss + args.lam * CMA_text.mean())/2 
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --is-train --prompt-name <prompt> --lam 0.001 --epochs 10 --bs 512 --lr 1e-5
```

## Inference
This inference takes approximately 1,000 seconds.

If you want fast inference, you can precompute and store the features beforehand.

You can download our model's checkpoint using the provided [link](https://drive.google.com/drive/folders/1k6trOT-zeVsT9WfbvavMPh2wwNBAFReK?usp=share_link).

```python
python inference.py --ckpt ./ckpt/CMA-ckpt.pt
```

## Citation

An update is coming soon.

## Thanks to

Our code is heavily based on 
- CLIP : https://github.com/openai/CLIP
- FLYP :https://github.com/locuslab/FLYP
- MCM : https://github.com/deeplearning-wisc/MCM
- NegLabel : https://github.com/XueJiang16/NegLabel
  
We thank the authors for open sourcing their code.
