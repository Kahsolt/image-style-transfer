#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/13

# 使用 aekl 进行风格迁移 (This not work... 🤔)
# migrated from https://github.com/gsurma/style_transfer

from utils import *

from torch.autograd import grad
from torch import Size
from torch.nn import Module as Model
import torchvision.transforms.functional as TF
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.autoencoders.vae import Encoder, DiagonalGaussianDistribution
from scipy.optimize import fmin_l_bfgs_b

OUTPUT_FILE = OUT_PATH / 'aekl.png'

VAE_CKPTS = [
  "stabilityai/sd-vae-ft-ema",
  "stabilityai/sd-vae-ft-mse",
  "stabilityai/sdxl-vae",
]


def load_img(fp:Path, resize:tuple=None) -> Tensor:
  img = Image.open(fp).convert(mode='RGB')
  if resize: img = img.resize(resize, Image.BILINEAR)
  X = TF.to_tensor(img)
  return X

def resize_match(dst:Tensor, src:Tensor) -> Tensor:
  C, H, W = dst.shape
  return TF.resize(src, (H, W), interpolation=TF.InterpolationMode.BILINEAR)


def Encoder_forward_hijack(self:Encoder, x:Tensor) -> Tuple[Tensor, List[Tensor]]:
  hs = []
  x = self.conv_in(x)
  hs.append(x)
  # down
  for down_block in self.down_blocks:
    x = down_block(x)
    hs.append(x)
  # middle
  x = self.mid_block(x)
  hs.append(x)
  # post-process
  x = self.conv_norm_out(x)
  hs.append(x)
  x = self.conv_act(x)
  x = self.conv_out(x)
  hs.append(x)
  return x, hs

def AutoencoderKL_encode_hijack(self:AutoencoderKL, x:Tensor, sample_posterior:bool=False) -> List[Tensor]:
  x, hs = Encoder_forward_hijack(self.encoder, x)
  moments = self.quant_conv(x)
  posterior = DiagonalGaussianDistribution(moments)
  if sample_posterior:
    z = posterior.sample()
  else:
    z = posterior.mode()
  hs.append(z)
  return hs


class Evaluator:

  def __init__(self, args, model:Model, shape:Size, hs_c:List[Tensor], hs_s:List[Tensor]):
    self.args = args
    self.model = model
    self.shape = shape
    self.hs_c = hs_c
    self.hs_s = hs_s
    self.content_layers: List[int] = args.content_layers
    self.style_layers: List[int] = args.style_layers
    self.w_loss_c: float = args.loss_c
    self.w_loss_s: float = args.loss_s
    self.w_loss_v: float = args.loss_v
    self.w_loss_vf: float = args.loss_vf

  @property
  def img_h(self): return self.shape[0]
  @property
  def img_w(self): return self.shape[1]
  @property
  def img_c(self): return self.shape[2]

  def content_loss(self, content:Tensor, x:Tensor) -> Tensor:
    return torch.mean(torch.square(x - content))   # L2

  def style_loss(self, style:Tensor, x:Tensor) -> Tensor:
    def gram_matrix(x:Tensor) -> Tensor:
      B, C, H, W = x.shape
      assert B == 1
      fmap = x.view(C, H * W)
      return torch.mm(fmap, fmap.T)
    style = gram_matrix(style)
    x = gram_matrix(x)
    size = self.img_h * self.img_w
    return torch.mean(torch.square(style - x)) / (4.0 * (self.img_c ** 2) * (size ** 2))

  def total_variation_loss(self, x:Tensor) -> Tensor:
    a = torch.square(x[:, :self.img_h-1, :self.img_w-1, :] - x[:, 1:, :self.img_w-1, :])
    b = torch.square(x[:, :self.img_h-1, :self.img_w-1, :] - x[:, :self.img_h-1, 1:, :])
    return torch.mean(torch.pow(a + b, self.w_loss_vf))

  @torch.enable_grad()
  def evaluate_loss_and_gradients(self, x:ndarray) -> Tuple[float, ndarray]:
    X = torch.from_numpy(x).float().reshape(self.shape)
    X = X.unsqueeze(dim=0).detach().clone().to(device)
    X.requires_grad = True
    hs = AutoencoderKL_encode_hijack(self.model, X)

    # losses
    loss_c_list = [self.content_loss(self.hs_c[idx], hs[idx]) for idx in self.content_layers]
    loss_s_list = [self.style_loss  (self.hs_s[idx], hs[idx]) for idx in self.style_layers  ]
    loss_c = sum(loss_c_list) / len(loss_c_list)
    loss_s = sum(loss_s_list) / len(loss_s_list)
    loss_v = self.total_variation_loss(X)
    loss = loss_c * self.w_loss_c \
         + loss_s * self.w_loss_s \
         + loss_v * self.w_loss_v
    loss = loss / 1000
    print(f'>> loss: {loss:.5f}, loss_c: {loss_c:.5f}, loss_s: {loss_s:.5f}, loss_v: {loss_v:.5f}')

    # gradients
    gradients = grad(loss, X, grad_outputs=loss)[0]

    return loss.item(), gradients.flatten().cpu().numpy().astype(np.float64)

  def loss(self, x):
    loss, gradients = self.evaluate_loss_and_gradients(x)
    self._gradients = gradients
    return loss

  def gradients(self, x):
    return self._gradients


@torch.no_grad()
def run(args):
  model: AutoencoderKL = AutoencoderKL.from_pretrained(args.repo)
  model = model.eval().to(device)

  resize = [int(x) for x in args.resize.split(',')] if args.resize else None
  X_c = load_img(args.content, resize)
  X_s = load_img(args.style)
  X_s = resize_match(X_c, X_s)

  hs_c = AutoencoderKL_encode_hijack(model, X_c.unsqueeze(dim=0).to(device))
  hs_s = AutoencoderKL_encode_hijack(model, X_s.unsqueeze(dim=0).to(device))
  evaluator = Evaluator(args, model, X_c.shape, hs_c, hs_s)

  x = X_c + torch.rand_like(X_c).numpy() * 0.1
  bounds = [(0.0, 1.0)] * X_c.numel()
  for i in range(args.n_iters):
    x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, bounds=bounds, maxfun=args.n_iters*2)
    print(f'[Iter {i}] loss: {loss}')
  x = x.reshape(X_c.shape).transpose([1, 2, 0])
  x = np.clip(x, 0.0, 1.0)
  x = (x.astype(np.float32) * 255).astype(np.uint8)
  print(f'>> save to {args.output}')
  Image.fromarray(x).save(args.output)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-R', '--repo',    default=VAE_CKPTS[-1], help='huggingface AutoencoderKL model name, you can search from https://huggingface.co/models?other=diffusers%3AAutoencoderKL')
  parser.add_argument('-r', '--resize',  default='512,512',    type=str,   help='resize input image: w,h')
  parser.add_argument('-c', '--content', default=DEFAULT_CONTENT_FILE, type=Path,  help='path to content image file')
  parser.add_argument('-s', '--style',   default=DEFAULT_STYLE_FILE,   type=Path,  help='path to style image file')
  parser.add_argument('-o', '--output',  default=OUTPUT_FILE,  type=Path,  help='path to output image file')
  parser.add_argument('-C', '--content_layers', default=[7, 8],    nargs='+', type=int, help='layer index for content fmap')
  parser.add_argument('-S', '--style_layers',   default=[0, 1, 2], nargs='+', type=int, help='layer index for style fmap')
  parser.add_argument('-N', '--n_iters', default=10,           type=int,   help='number of optimization iterations')
  parser.add_argument('--loss_c',        default=0.02,         type=float, help='loss weight for content')
  parser.add_argument('--loss_s',        default=4.5,          type=float, help='loss weight for style')
  parser.add_argument('--loss_v',        default=0.995,        type=float, help='loss weight for total variation')
  parser.add_argument('--loss_vf',       default=1.25,         type=float, help='loss power factor for total variation')
  args, _ = parser.parse_known_args()

  run(args)
