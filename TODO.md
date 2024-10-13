### Todo List

⚪ 扩展 run_vgg19.py 脚本

- 将文件名中的 `19` 去掉，扩展支持命令行参数 `-M` 以选择模型，参考 `run_aekl.py` 中的实现
- 增加 demo: vgg11, vgg19_bn
  - 为每个模型配一套独立的 style_layers 和 content_layers 设置，自己试着调调参 ;)

⚪ 扩展至其他模型族

ℹ 每个模型对应一个新脚本，从 `run_vgg.py` 复制一份开始改

- CNN
  - alexnet (增加 demo: alexnet)
  - googlenet (增加 demo: googlenet)
  - inception (增加 demo: inception_v3)
  - resnet 系列 (增加 demo: resnet18 / resnet50)
  - densenet 系列 (增加 demo: densenet121)
  - convnext 系列 (增加 demo: convnext_tiny)
- CNN (light weight)
  - squeezenet 系列 (增加 demo: squeezenet1_1)
  - mobilenet 系列 (增加 demo: mobilenet_v2 / mobilenet_v3_small / mobilenet_v3_large)
  - shufflenet 系列 (增加 demo: shufflenet_v2_x0_5 / shufflenet_v2_x2_0)
  - mnasnet 系列 (增加 demo: mnasnet0_5 / mnasnet1_3)
- transformer
  - vit 系列 (增加 demo: vit_b_16)
  - swin 系列 (增加 demo: swin_t / swin_v2_t)
  - maxvit 系列 (增加 demo: maxvit_t)
