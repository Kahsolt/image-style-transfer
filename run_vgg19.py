#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/10 

# 使用 vgg19 进行风格迁移
# migrated from https://github.com/EliShayGH/deep-learning-style-transfer

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

from PIL import Image
from argparse import ArgumentParser
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torchvision import models as M, transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('>> device:', device)
dtype = torch.bfloat16
print('>> dtype:', dtype)

style_layers = {
  '0':  0.8,
  '5':  1.0,
  '10': 0.5,
  '19': 0.3,
  '28': 0.1,
}
content_layers = {
  '34': 1.0,
}
content_weight = 1   # alpha
style_weight = 1e6   # beta
steps = 1500         # decide how many iterations to update your image (5000)
log_every = 100
lr = 0.1             # ~12 in pixel value


def im_load(fp:str, resize:int=400, shape:Tuple[int, int]=None) -> Tensor:
  img = Image.open(fp).convert('RGB')
  size = shape or min(resize, max(img.size))
  transform = T.Compose([
    T.Resize(size),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
  ])
  return transform(img).unsqueeze(0)

def im_convert(x:Tensor):
  x = x.detach().squeeze().permute(1, 2, 0)
  mean = Tensor(IMAGENET_MEAN).to(device).unsqueeze(0).unsqueeze(0)
  std  = Tensor(IMAGENET_STD) .to(device).unsqueeze(0).unsqueeze(0)
  x = x * std + mean
  return x.clamp(0, 1).float().cpu().numpy()

def get_activations(x:Tensor, model:nn.Module) -> Dict[str, Tensor]:
  global style_layers, content_layers
  acts = {}
  for name, layer in model._modules.items():   # sequential forward
    x = layer(x)
    if name in style_layers or name in content_layers:
      acts[name] = x
  return acts

def gram_matrix(x:Tensor) -> Tensor:
  _, c, h, w = x.shape
  x = x.view(c, h * w)
  return torch.mm(x, x.T)


def run(c_fp:str, s_fp:str):
  vgg = M.vgg19(pretrained=True).features.eval().to(device, dtype)
  for param in vgg.parameters():
    param.requires_grad_(False)
  print(vgg)

  content = im_load(c_fp)                          .to(device, dtype)
  style   = im_load(s_fp, shape=content.shape[-2:]).to(device, dtype)
  content_features = get_activations(content, vgg)
  style_grams = {layer: gram_matrix(feature) for layer, feature in get_activations(style, vgg).items()}

  target = content.detach().clone().requires_grad_(True)
  optim = Adam([target], lr=lr)
  for i in tqdm(range(steps)):
    activations = get_activations(target, vgg)

    content_loss = 0.0
    for layer, weight in content_layers.items():
      act = activations[layer]
      content_loss += weight * F.mse_loss(act, content_features[layer])
    style_loss = 0.0
    for layer, weight in style_layers.items():
      act = activations[layer]
      style_loss += weight * F.mse_loss(gram_matrix(act), style_grams[layer]) / act[0].numel()
    loss = content_weight * content_loss + style_weight * style_loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (i + 1) % log_every == 0:
      print('>> loss:', loss.item(), 'content_loss', content_loss.item(), 'style_loss:', style_loss.item())

  imgs = [content, target, style]
  titles = ['content', 'transfered', 'style']
  nfig = len(imgs)
  fig, axs = plt.subplots(1, nfig, figsize=(14, 4))
  for i, ax in enumerate(axs):
    ax.imshow(im_convert(imgs[i]))
    ax.set_title(titles[i])
  plt.suptitle('style-transfer via vgg19')
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-C', default='./img/Tuebingen_Neckarfront.jpg', help='content image file')
  parser.add_argument('-S', default='./img/vangogh_starry_night.jpg',  help='style image file')
  args = parser.parse_args()

  run(args.C, args.S)
