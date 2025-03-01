import numpy as np
import clip
from tqdm import tqdm


import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import  MNIST, CIFAR10, SVHN, CIFAR100, ImageFolder
import torchvision.transforms as transforms
import torchvision

import argparse

from torch.autograd import Variable

from utils.common import setup_seed
from utils.metrics import get_measures
from utils.load_model import set_model_clip
from utils.loader import test_loader, train_loader
from utils.train_utils import train

from nltk.corpus import wordnet

import random


def process_args():

    parser = argparse.ArgumentParser(description = 'Training CLIP Out-of-distribution Detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--models', type=str, default='ViT-B/16')
    parser.add_argument('--clip', type=str, default='openai')
    parser.add_argument('--ckpt', type=str, default='./')
    parser.add_argument('--ood-dataset',type=str, default='iNaturalist')
    parser.add_argument('--methods', type=str, default='CMA')
    parser.add_argument('--benchmark',type=str, default='imagenet')
    parser.add_argument('--dir', type=str, default='./')
    parser.add_argument('--seed', type=int, default=0)    
    parser.add_argument('--is-train', default=False,action="store_true")
    parser.add_argument('--prompt-name', type=str, default='a photo of a')
    
    parser.add_argument('--multiprocessing-distributed', default=False,action="store_true")
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    
    parser.add_argument('--lam', type=float, default=0.0)
    
    args = parser.parse_args()
    
    
    return args

def main():
    args = process_args()
    setup_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = set_model_clip(args)
    
    model.to(device)
    
    in_dataloader, texts_in = train_loader(args, preprocess, device)
    model.train()
    train(args, model, in_dataloader, texts_in, device)
    
if __name__ == '__main__':
    main()

    
