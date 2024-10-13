#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/13 

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)
import warnings ; warnings.filterwarnings(action='ignore', category=FutureWarning)

from PIL import Image
from pathlib import Path
from time import time
from argparse import ArgumentParser
from typing import List, Tuple, Dict, NamedTuple, Union, Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms as T
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from tqdm import tqdm

npimg = ndarray
Normalizer = Union[
  Literal['norm'],    # clf models
  Literal['±1'],      # vae models
]

BASE_PATH = Path(__file__).parent
IMG_PATH = BASE_PATH / 'img'
DEFAULT_CONTENT_FILE = IMG_PATH / 'Tuebingen_Neckarfront.jpg'
DEFAULT_STYLE_FILE = IMG_PATH / 'vangogh_starry_night.jpg'
OUT_PATH = BASE_PATH / 'out' ; OUT_PATH.mkdir(exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('>> device:', device)
dtype = torch.bfloat16
print('>> dtype:', dtype)


def im_load(fp:str, resize:int=400, shape:Tuple[int, int]=None, normalizer:Normalizer='norm') -> Tensor:
  img = Image.open(fp).convert('RGB')
  size = shape or min(resize, max(img.size))
  if normalizer == 'norm':
    norm_fn = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
  elif normalizer == '±1':
    norm_fn = T.Lambda(lambda x: x * 2 - 1)
  else:
    norm_fn = T.Lambda(lambda x: x)
  transform = T.Compose([
    T.Resize(size),
    T.ToTensor(),
    norm_fn
  ])
  return transform(img).unsqueeze(0)

def im_convert(x:Tensor, normalizer:Normalizer='norm') -> npimg:
  x = x.detach().squeeze().permute(1, 2, 0)
  if normalizer == 'norm':
    mean = Tensor(IMAGENET_MEAN).to(device).unsqueeze(0).unsqueeze(0)
    std  = Tensor(IMAGENET_STD) .to(device).unsqueeze(0).unsqueeze(0)
    x = x * std + mean
  elif normalizer == '±1':
    x = (x + 1) / 2
  x = x.clamp(0, 1).float().cpu().numpy()
  return x

def im_save(x:Tensor, model:str, normalizer:Normalizer='norm'):
  img = Image.fromarray((im_convert(x, normalizer) * 255).astype(np.uint8))
  fp = OUT_PATH / f'{model}.jpg'
  print(f'>> savefig {fp}')
  img.save(fp)

def im_show_compare(imgs:List[Tensor], model:str, normalizer:Normalizer='norm'):
  titles = ['content', 'transfered', 'style']
  nfig = len(imgs) ; assert nfig == len(titles)
  plt.clf()
  fig, axs = plt.subplots(1, nfig, figsize=(14, 4))
  for i, ax in enumerate(axs):
    ax.imshow(im_convert(imgs[i], normalizer))
    ax.set_title(titles[i])
  plt.suptitle(f'style-transfer via {model}')
  plt.tight_layout()
  plt.show()
  plt.close()


def gram_matrix(x:Tensor) -> Tensor:
  _, c, h, w = x.shape
  x = x.view(c, h * w)
  return torch.mm(x, x.T)
