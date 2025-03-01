
import torch
import torch.nn.functional as F
import numpy as np
from utils.metrics import get_measures

import numpy as np
import clip
from tqdm import tqdm
import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import  MNIST, CIFAR10, SVHN, CIFAR100, ImageFolder
import torchvision.transforms as transforms
import torchvision

from prompt import Prompt_classes

import argparse

from torch.autograd import Variable

from utils.common import setup_seed
from utils.metrics import get_measures
from utils.load_model import set_model_clip
from utils.loader import test_loader_list_MOS

import nltk
from nltk.corpus import wordnet as wn
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description = 'Evaluates CLIP Out-of-distribution Detection',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--models', type=str, default='ViT-B/16')
parser.add_argument('--clip', type=str, default='openai')
parser.add_argument('--ckpt', type=str, default='./')
parser.add_argument('--ood-dataset',type=str, default='iNaturalist')
parser.add_argument('--methods', type=str, default='flyp')
parser.add_argument('--benchmark',type=str, default='imagenet')
parser.add_argument('--prompt-name',type=str, default='The nice')
parser.add_argument('--dir', type=str, default='/data')
parser.add_argument('--bs', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sim', type=int, default=1.0)
parser.add_argument('--is-train', default=False,action="store_true")


parser.add_argument('--multiprocessing-distributed', default=False,action="store_true")

args = parser.parse_args()


## Load Model ##
print('model load !')
model, _, preprocess = set_model_clip(args)
in_dataloader, out_dataloader, texts_in = test_loader_list_MOS(args, preprocess, device)
imagenet_classes,_ = Prompt_classes('imagenet')
if args.prompt_name == 'The nice':
    print('Prompt -The nice-')
    texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes]).to(device)
elif args.prompt_name == 'a photo of a':
    print('Prompt -A photo of a-')
    texts_in = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes]).to(device)
elif args.prompt_name == 'no':
    print('Prompt -no-')
    texts_in = clip.tokenize([f"{c}" for c in imagenet_classes]).to(device)
# texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes]).to(device)
# texts_in = clip.tokenize([f"{c}" for c in imagenet_classes]).to(device)
model.to(device)
model = model.eval()
print('model load finished !')
################

# ###########################################################
## imagenet text token -> imagenet text embedding ##
print('imagenet text-embedding !')
with torch.no_grad():
    imagenet_texts = model.module.encode_text(texts_in)
    imagenet_texts_cpu = imagenet_texts.cpu()
    
imagenet_texts_cpu_norm = imagenet_texts_cpu / imagenet_texts_cpu.norm(dim=-1,keepdim=True)
print('imagenet text-embedding finished!')
# ####################################################

n = np.load('./Neglabel/neg_label_10000.npy')

print('neg_text-embedding !')
with torch.no_grad():
    neg_text = model.module.encode_text(clip.tokenize([i for i in n]).to(device))

print('neg_text-embedding finished!')

neg_text = neg_text / neg_text.norm(dim=-1, keepdim=True)
    
# ##############################################################
print('imagenet features !')
encoded_images = []
tqdm_object = tqdm(in_dataloader, total=len(in_dataloader))
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm_object):
        bz = images.size(0)
        images = images.cuda()
        if args.ckpt == './':
            image_embeddings = model.module.encode_image(images)    
        else:
            image_embeddings = model.module.encode_image(images)
        image_embeddings_cpu = image_embeddings.cpu()
        encoded_images.append(image_embeddings_cpu)

imagenet_images = torch.cat(encoded_images)
imagenet_images_norm = imagenet_images / imagenet_images.norm(dim=-1,keepdim=True)

# ################################################################
print('ood features !')

ood_names = ["iNaturalist","SUN", "Places", "Textures"]

for i, (name, loader) in enumerate(zip(ood_names,out_dataloader)):
    print(name)
    tqdm_object = tqdm(loader, total=len(loader))
    ood_images_feature=[]
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            bz = images.size(0)
            images = images.cuda()
            if args.ckpt == './':
                image_embeddings = model.module.encode_image(images)    
            else:
                image_embeddings = model.module.encode_image(images)
            image_embeddings_cpu = image_embeddings.cpu()
            if name == 'iNaturalist':
                ood_images_feature.append(image_embeddings_cpu)
            if name == 'SUN':
                ood_images_feature.append(image_embeddings_cpu)
            if name == 'Places':
                ood_images_feature.append(image_embeddings_cpu)
            if name == 'Textures':
                ood_images_feature.append(image_embeddings_cpu)
        ood_images_feature = torch.cat(ood_images_feature)
        ood_images_feature /= ood_images_feature.norm(dim=-1,keepdim=True)
    if name == 'iNaturalist':
        inaturalist_images = ood_images_feature
    if name == 'SUN':
        sun_images = ood_images_feature
    if name == 'Places':
        places_images = ood_images_feature
    if name == 'Textures':
        textures_images = ood_images_feature

to_np = lambda x : x.data.numpy()    

imagenet_logits = imagenet_images_norm @ imagenet_texts_cpu_norm.T
inaturalist_logits = inaturalist_images @ imagenet_texts_cpu_norm.T
sun_logits = sun_images @ imagenet_texts_cpu_norm.T
places_logits = places_images @ imagenet_texts_cpu_norm.T
textures_logits = textures_images @ imagenet_texts_cpu_norm.T

labels = torch.load('./CLIP_im1k_features/val/valtarget.pt')

acc = (np.argmax(imagenet_logits.cpu().detach().numpy(),axis=1)== labels.cpu().detach().numpy()).sum()
print(f"ACC : {acc/50000} !")



imagenet_Negative_logits = imagenet_images_norm @ neg_text.T.cpu()
inaturalist_Negative_logits = inaturalist_images @ neg_text.T.cpu()
sun_Negative_logits = sun_images @ neg_text.T.cpu()
places_Negative_logits = places_images @ neg_text.T.cpu()
textures_Negative_logits = textures_images @ neg_text.T.cpu()

imagenet_random_logits = torch.cat([imagenet_logits,imagenet_Negative_logits],dim=1)/0.01
inaturalist_random_logits = torch.cat([inaturalist_logits,inaturalist_Negative_logits],dim=1)/0.01
sun_random_logits = torch.cat([sun_logits,sun_Negative_logits],dim=1)/0.01
places_random_logits = torch.cat([places_logits,places_Negative_logits],dim=1)/0.01
textures_random_logits = torch.cat([textures_logits,textures_Negative_logits],dim=1)/0.01


imagenet_random_logits_np = to_np(F.softmax(imagenet_random_logits,dim=1))
inaturalist_random_logits_np = to_np(F.softmax(inaturalist_random_logits,dim=1))
sun_random_logits_np = to_np(F.softmax(sun_random_logits,dim=1))
places_random_logits_np = to_np(F.softmax(places_random_logits,dim=1))
textures_random_logits_np = to_np(F.softmax(textures_random_logits,dim=1))

imagenet_in_random_logits = np.sum(imagenet_random_logits_np[:,:1000],axis=1)
inaturalist_in_random_logits = np.sum(inaturalist_random_logits_np[:,:1000],axis=1)
sun_in_random_logits = np.sum(sun_random_logits_np[:,:1000],axis=1)
places_in_random_logits = np.sum(places_random_logits_np[:,:1000],axis=1)
textures_in_random_logits = np.sum(textures_random_logits_np[:,:1000],axis=1)

imagenet_in_random_logits = imagenet_in_random_logits.reshape(-1,1)
inaturalist_in_random_logits = inaturalist_in_random_logits.reshape(-1,1)
sun_in_random_logits = sun_in_random_logits.reshape(-1,1)
places_in_random_logits = places_in_random_logits.reshape(-1,1)
textures_in_random_logits = textures_in_random_logits.reshape(-1,1)

print('NegLabel !')
print("inaturalist", get_measures(imagenet_in_random_logits,inaturalist_in_random_logits))
print("sun", get_measures(imagenet_in_random_logits, sun_in_random_logits))
print("places", get_measures(imagenet_in_random_logits,places_in_random_logits))
print("textures", get_measures(imagenet_in_random_logits,textures_in_random_logits))

imagenet_logits_np = to_np(F.softmax(imagenet_logits,dim=1))
inaturalist_logits_np = to_np(F.softmax(inaturalist_logits,dim=1))
sun_logits_np = to_np(F.softmax(sun_logits,dim=1))
places_logits_np = to_np(F.softmax(places_logits,dim=1))
textures_logits_np = to_np(F.softmax(textures_logits,dim=1))

MCM_imagenet = np.max(imagenet_logits_np,axis=1).reshape(-1,1)
MCM_inaturalist = np.max(inaturalist_logits_np,axis=1).reshape(-1,1)
MCM_sun = np.max(sun_logits_np,axis=1).reshape(-1,1)
MCM_places = np.max(places_logits_np,axis=1).reshape(-1,1)
MCM_textures = np.max(textures_logits_np,axis=1).reshape(-1,1)

print('MCM !')
print("inaturalist", get_measures(MCM_imagenet,MCM_inaturalist))
print("sun", get_measures(MCM_imagenet,MCM_sun))
print("places", get_measures(MCM_imagenet,MCM_places))
print("textures", get_measures(MCM_imagenet,MCM_textures))

