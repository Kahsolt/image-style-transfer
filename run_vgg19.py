#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/10 

# 使用 vgg19 进行风格迁移
# migrated from https://github.com/EliShayGH/deep-learning-style-transfer

from utils import *
from torchvision import models as M


def get_activations(x:Tensor, model:nn.Module) -> Dict[str, Tensor]:
  global style_layers, content_layers
  acts = {}
  for name, layer in model._modules.items():   # sequential forward
    x = layer(x)
    if name in style_layers or name in content_layers:
      acts[name] = x
  return acts


def run(c_fp:str, s_fp:str):
  ''' Model '''
  model = 'vgg19'
  print(f'>> model vgg19')
  vgg = M.vgg19(pretrained=True).features.eval().to(device, dtype)
  for param in vgg.parameters():
    param.requires_grad_(False)
  print(vgg)

  ''' Make GT '''
  content = im_load(c_fp)                          .to(device, dtype)
  style   = im_load(s_fp, shape=content.shape[-2:]).to(device, dtype)
  content_features = get_activations(content, vgg)
  style_grams = {layer: gram_matrix(feature) for layer, feature in get_activations(style, vgg).items()}

  ''' Optimize! '''
  target = content.detach().clone().requires_grad_(True)
  optim = Adam([target], lr=lr)

  ts_start = time()
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

    if (i + 1) % 100 == 0:
      print('>> loss:', loss.item(), 'content_loss', content_loss.item(), 'style_loss:', style_loss.item())

  ts_end = time()
  print(f'>> Runtime: {ts_end - ts_start:.3f}s')

  ''' Show '''
  im_save(target, model)
  #im_show_compare([content, target, style], model)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-C', default=DEFAULT_CONTENT_FILE, help='content image file')
  parser.add_argument('-S', default=DEFAULT_STYLE_FILE, help='style image file')
  args = parser.parse_args()

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
  lr = 0.1             # ~12 in pixel value

  run(args.C, args.S)
