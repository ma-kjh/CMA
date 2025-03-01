

import torch
import numpy as np
import clip
from torch.utils.data import DataLoader, Dataset

import webdataset as wds
import torch
from PIL import Image
from torchvision import transforms

import torchvision

from prompt import Prompt_classes

from torchvision import datasets, transforms


def train_loader(args, preprocess, device=None):    
    
    if args.benchmark == 'imagenet':
        imagenet_classes,_ = Prompt_classes('imagenet')
        imagenet_train = datasets.ImageFolder(root=f'{args.dir}/imagenet_1k/train', transform=preprocess)
        in_dataloader = DataLoader(imagenet_train, shuffle=True, batch_size = args.bs, num_workers=32)    
        
    with torch.no_grad():
        if args.prompt_name == 'The nice':
            print("Prompt is -The nice-")
            texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes])
        elif args.prompt_name == 'a photo of a':
            print("Prompt is -a photo of a-")
            texts_in = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes])
        else:
            print("Prompt is -no-")
            texts_in = clip.tokenize([f"{c}" for c in imagenet_classes])
        
    if args.multiprocessing_distributed:
        return in_dataloader, train_sampler, texts_in
    else:
        return in_dataloader, texts_in

def test_loader(args, preprocess, device):    
    
    if args.benchmark == 'imagenet':
        imagenet_classes,_ = Prompt_classes('imagenet')
        imagenet_test = datasets.ImageFolder(root=f'{args.dir}/imagenet_1k/val', transform=preprocess)
        
        if args.ood_dataset == 'iNaturalist':
            ood_dataset = datasets.ImageFolder(root=f'{args.dir}/MOS/iNaturalist', transform=preprocess)
        elif args.ood_dataset == 'SUN':
            ood_dataset = datasets.ImageFolder(root=f'{args.dir}/MOS/SUN', transform=preprocess)
        elif args.ood_dataset == 'Places':
            ood_dataset = datasets.ImageFolder(root=f'{args.dir}/MOS/Places', transform=preprocess)
        elif args.ood_dataset == 'dtd':
            dtd = ood_dataset = datasets.ImageFolder(root=f'{args.dir}/MOS/dtd/images', transform=preprocess)
        
        in_dataloader = DataLoader(imagenet_test, shuffle=False, batch_size = args.bs, num_workers=32)       
        out_dataloader = DataLoader(ood_dataset, shuffle=False, batch_size =args.bs, num_workers=32)
        
    with torch.no_grad():
        texts_in = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes]).to(device)
    
    return in_dataloader, out_dataloader, texts_in

def test_loader_list_MOS(args, preprocess, device):    
    
    if args.benchmark == 'imagenet':
        imagenet_classes,_ = Prompt_classes('imagenet')
        imagenet_test = datasets.ImageFolder(root=f'{args.dir}/imagenet_1k/val', transform=preprocess)
        
        
        iNaturalist = datasets.ImageFolder(root=f'{args.dir}/MOS/iNaturalist', transform=preprocess)
        SUN = datasets.ImageFolder(root=f'{args.dir}/MOS/SUN', transform=preprocess)
        Places = datasets.ImageFolder(root=f'{args.dir}/MOS/Places', transform=preprocess)
        dtd = datasets.ImageFolder(root=f'{args.dir}/MOS/dtd/images', transform=preprocess)
        
        in_dataloader = DataLoader(imagenet_test, shuffle=False, batch_size = args.bs, num_workers=32)
        
        iNaturalist = DataLoader(iNaturalist, shuffle=False, batch_size= args.bs, num_workers=32)
        SUN = DataLoader(SUN, shuffle=False, batch_size= args.bs, num_workers=32)
        Places = DataLoader(Places, shuffle=False, batch_size= args.bs, num_workers=32)
        dtd = DataLoader(dtd, shuffle=False, batch_size= args.bs, num_workers=32)
        
        out_dataloader = [iNaturalist, SUN, Places, dtd]
    
    with torch.no_grad():
        texts_in = clip.tokenize([f"{args.prompt_name} {c}" for c in imagenet_classes]).to(device)
    
    return in_dataloader, out_dataloader, texts_in

def test_loader_list_OpenOOD(args, preprocess, device):    
    
    if args.benchmark == 'imagenet':
        imagenet_classes,_ = Prompt_classes('imagenet')
        imagenet_test = image_title_dataset(f'{args.dir}/openood/benchmark_imglist/imagenet/test_imagenet.txt',transforms= preprocess)
        
        ssb = image_title_dataset(f'{args.dir}/openood/benchmark_imglist/imagenet/test_ssb_hard.txt',transforms= preprocess)
        ninco = image_title_dataset(f'{args.dir}/openood/benchmark_imglist/imagenet/test_ninco.txt',transforms= preprocess)

        iNaturalist = image_title_dataset(f'{args.dir}/openood/benchmark_imglist/imagenet/test_inaturalist.txt',transforms= preprocess)
        textures = image_title_dataset(f'{args.dir}/openood/benchmark_imglist/imagenet/test_textures.txt',transforms= preprocess)
        openimage_o = image_title_dataset(f'{args.dir}/openood/benchmark_imglist/imagenet/test_openimage_o.txt',transforms= preprocess)
        
        in_dataloader = DataLoader(imagenet_test, shuffle=False, batch_size = args.bs, num_workers=32)
        
        ssb = DataLoader(ssb, shuffle=False, batch_size= args.bs, num_workers=32)
        ninco = DataLoader(ninco, shuffle=False, batch_size= args.bs, num_workers=32)
        
        iNaturalist = DataLoader(iNaturalist, shuffle=False, batch_size= args.bs, num_workers=32)
        textures = DataLoader(textures, shuffle=False, batch_size= args.bs, num_workers=32)
        openimage_o = DataLoader(openimage_o, shuffle=False, batch_size= args.bs, num_workers=32)
        
        out_dataloader = [ssb, ninco, iNaturalist, textures, openimage_o]
    
    with torch.no_grad():
        # texts_in = clip.tokenize([f"The nice {c}" for c in imagenet_classes]).to(device)
        # texts_in = clip.tokenize([f"a photo of a {c}" for c in imagenet_classes]).to(device)
        texts_in = clip.tokenize([f"{c}" for c in imagenet_classes]).to(device)
    
    return in_dataloader, out_dataloader, texts_in


class image_title_dataset(Dataset):
    def __init__(self, annotation='', transforms = None):
        
        self.transforms = transforms
        self.annotation = annotation
        files = pd.read_csv(annotation, sep=' ',header=None)
        self.data = files[0]
        self.targets = files[1]

    def __len__(self):
        return len(self.data)
    
    def classes(self):
        return self.targets

    def __getitem__(self, idx):
        if self.annotation == f'{args.dir}/openood/benchmark_imglist/imagenet/test_textures.txt':
            self.image = self.transforms(Image.open(f'{args.dir}/openood/images_classic/'+self.data[idx]).convert('RGB'))
        else:
            self.image = self.transforms(Image.open(f'{args.dir}/openood/images_largescale/'+self.data[idx]).convert('RGB'))
        return self.image, self.targets[idx]

