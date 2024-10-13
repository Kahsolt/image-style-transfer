#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/10 

# 使用 aekl/tae 进行风格迁移

from utils import *
from diffusers.models import AutoencoderKL, AutoencoderTiny

AutoEncoder = type[Union[AutoencoderKL, AutoencoderTiny]]

class Entry(NamedTuple):
  path: str
  cls: AutoEncoder

PRETRAINED_MODELS = {
  'sd-vae-ft-ema': Entry('stabilityai/sd-vae-ft-ema',     AutoencoderKL),
  'sd-vae-ft-mse': Entry('stabilityai/sd-vae-ft-mse',     AutoencoderKL),
  'sdxl-vae':      Entry('stabilityai/sdxl-vae',          AutoencoderKL),
  'taesd':         Entry('madebyollin/taesd',             AutoencoderTiny),
  'taesd-x4':      Entry('madebyollin/taesd-x4-upscaler', AutoencoderTiny),
  'taesdxl':       Entry('madebyollin/taesdxl',           AutoencoderTiny),
  'taesd3':        Entry('madebyollin/taesd3',            AutoencoderTiny),
}


def AutoencoderKL_encode_hijack(self:AutoencoderKL, x:Tensor) -> Dict[str, Tensor]:
  acts = {}
  x = self.encoder.conv_in(x)
  acts['conv_in'] = x
  for idx, down_block in enumerate(self.encoder.down_blocks):
    x = down_block(x)
    acts[f'down_block.{idx}'] = x
  x = self.encoder.mid_block(x)
  acts['mid_block'] = x
  x = self.encoder.conv_norm_out(x)
  x = self.encoder.conv_act(x)
  x = self.encoder.conv_out(x)
  acts['conv_out'] = x
  if VAE_POSTPROCESS:
    x = self.quant_conv(x)
    x = torch.chunk(x, 2, dim=1)[0]   # DiagonalGaussianDistribution
    acts['quant'] = x
  return acts

def AutoencoderTiny_encode_hijack(self:AutoencoderTiny, x:Tensor) -> Dict[str, Tensor]:
  global style_layers, content_layers
  acts = {}
  for name, layer in self.encoder.layers._modules.items():
    x = layer(x)
    if name in style_layers or name in content_layers:
      acts[name] = x
  if VAE_POSTPROCESS:
    x = self.scale_latents(x).mul_(255).round_().byte()
    x = self.unscale_latents(x / 255.0)
    acts['quant'] = x
  return acts

def get_activations(x:Tensor, model:nn.Module) -> Dict[str, Tensor]:
  if isinstance(model, AutoencoderKL):
    return AutoencoderKL_encode_hijack(model, x)
  if isinstance(model, AutoencoderTiny):
    return AutoencoderTiny_encode_hijack(model, x)


def run(model:str, c_fp:str, s_fp:str):
  ''' Model '''
  entry = PRETRAINED_MODELS[model]
  print(f'>> model repo {entry.path!r}')
  vae: AutoEncoder = entry.cls.from_pretrained(entry.path)
  del vae.decoder
  vae = vae.eval().to(device, dtype)
  for param in vae.parameters():
    param.requires_grad_(False)
  print(vae)

  ''' Make GT '''
  content = im_load(c_fp,                           normalizer='±1').to(device, dtype)
  style   = im_load(s_fp, shape=content.shape[-2:], normalizer='±1').to(device, dtype)
  content_features = get_activations(content, vae)
  style_grams = {layer: gram_matrix(feature) for layer, feature in get_activations(style, vae).items()}

  ''' Optimize! '''
  target = content.detach().clone().requires_grad_(True)
  optim = Adam([target], lr=lr)

  ts_start = time()
  for i in tqdm(range(steps)):
    activations = get_activations(target, vae)

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
  im_save(target, model, normalizer='±1')
  #im_show_compare([content, target, style], model, normalizer='±1')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', default='taesdxl', choices=PRETRAINED_MODELS.keys())
  parser.add_argument('-C', default=DEFAULT_CONTENT_FILE, help='content image file')
  parser.add_argument('-S', default=DEFAULT_STYLE_FILE, help='style image file')
  args = parser.parse_args()

  VAE_POSTPROCESS = False
  if args.M.startswith('taesd'):  # preset for taesdxl
    style_layers = {
      '0': 0.75,
      '3': 1.0,
      '7': 0.5,
      '11': 0.3,
    }
    content_layers = {
      '13': 1.0,
      #'quant': 1.0,
    }
  elif args.M.startswith('sd'):   # preset for sdxl
    style_layers = {
      'conv_in': 0.75,
      'down_block.0': 1.0,
      'down_block.1': 0.4,
      'down_block.2': 0.2,
      'down_block.3': 0.1,
      #'mid_block': 0.1,
    }
    content_layers = {
      'conv_out': 1.0,
      #'quant': 1.0,
    }
  content_weight = 1   # alpha
  style_weight = 10    # beta
  steps = 1500         # decide how many iterations to update your image (5000)
  lr = 8 / 255

  run(args.M, args.C, args.S)
