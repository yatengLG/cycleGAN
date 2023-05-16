# 基于pytorch的cycleGANk复现

[cycleGAN官方代码](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

# 介绍

本复现项目中，使用resnet_9blocks作为生成器模型，使用3层的NLayerDiscriminator作为判别器模型。

训练相关参数与流程，均包含在cycle_gan.py文件中，总体不超200行，简单易懂，可读性极强。

调用代码为transfer.py，只需载入模型并指定图片，即可完成调用。

# 效果

| 原图                                                             | 官方                                                             | 复现                                                           |
|----------------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------------------|
| ![n02381460_1630_real.jpg](./example/n02381460_1630_real.jpg)  | ![n02381460_1630_fake.jpg](./example/n02381460_1630_fake.jpg)  | ![n02381460_1630_our.jpg](./example/n02381460_1630_our.jpg)  |
| ![n02381460_3330_real.jpg](./example/n02381460_3330_real.jpg)  | ![n02381460_3330_fake.jpg](./example/n02381460_3330_fake.jpg)  | ![n02381460_3330_our.jpg](./example/n02381460_3330_our.jpg)  |
| ![n02381460_7230_real.jpg](./example/n02381460_7230_real.jpg)  | ![n02381460_7230_fake.jpg](./example/n02381460_7230_fake.jpg)  | ![n02381460_7230_our.jpg](./example/n02381460_7230_our.jpg)  |
| ![n02391049_3270_real.jpg](./example/n02391049_3270_real.jpg)  | ![n02391049_3270_fake.jpg](./example/n02391049_3270_fake.jpg)  | ![n02391049_3270_our.jpg](./example/n02391049_3270_our.jpg)  |
| ![n02391049_5240_real.jpg](./example/n02391049_5240_real.jpg)  | ![n02391049_5240_fake.jpg](./example/n02391049_5240_fake.jpg)  | ![n02391049_5240_our.jpg](./example/n02391049_5240_our.jpg)  |
| ![n02391049_5670_real.jpg](./example/n02391049_5670_real.jpg)  | ![n02391049_5670_fake.jpg](./example/n02391049_5670_fake.jpg)  | ![n02391049_5670_our.jpg](./example/n02391049_5670_our.jpg)  |

# 训练

```shell
conda create -n cycleGAN python=3.8
conda activate cycleGAN
pip install -r requirements.txt

# 修改cycle_gan.py中213行，图片目录
python cycle_gan.py
```

你也可以下载本项目训练好的[生成器模型](https://github.com/yatengLG/cycleGAN/releases/tag/1.0.0)
# 损失

| G_loss                             | D_loss                            | GAN_loss                               | cycle_loss                                 | idt_loss                              |
|------------------------------------|-----------------------------------|----------------------------------------|--------------------------------------------|---------------------------------------|
| ![G_loss.png](example/G_loss.png)  | ![D_loss.png](example/D_loss.png) | ![gan_loss.png](example/gan_loss.png)  | ![cycle_loss.png](example/cycle_loss.png)  | ![idt_loss.png](example/idt_loss.png) |


# 过程

|              | EPOCH 0                                                           | EPOCH 30                                                               | EPOCH 40                                                              | EPOCH 50                                                              | EPOCH 80                                                              | EPOCH 120                                                               | EPOCH 160                                                               | EPOCH 200                                                               |
|--------------|-------------------------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| horse2zebra  | ![n02381460_4530_epoch_30.jpg](./example/n02381460_4530_real.jpg) | ![n02381460_4530_epoch_30.jpg](./example/n02381460_4530_epoch_30.jpg)  | ![n02381460_4530_epoch_40.jpg](./example/n02381460_4530_epoch_40.jpg) | ![n02381460_4530_epoch_50.jpg](./example/n02381460_4530_epoch_50.jpg) | ![n02381460_4530_epoch_80.jpg](./example/n02381460_4530_epoch_80.jpg) | ![n02381460_4530_epoch_120.jpg](./example/n02381460_4530_epoch_120.jpg) | ![n02381460_4530_epoch_160.jpg](./example/n02381460_4530_epoch_160.jpg) | ![n02381460_4530_epoch_200.jpg](./example/n02381460_4530_epoch_200.jpg) | 
| zebra2horse  | ![n02391049_3290_real.jpg](./example/n02391049_3290_real.jpg)     | ![n02391049_3290_epoch_30.jpg](./example/n02391049_3290_epoch_30.jpg)  | ![n02391049_3290_epoch_40.jpg](./example/n02391049_3290_epoch_40.jpg) | ![n02391049_3290_epoch_50.jpg](./example/n02391049_3290_epoch_50.jpg) | ![n02391049_3290_epoch_80.jpg](./example/n02391049_3290_epoch_80.jpg) | ![n02391049_3290_epoch_120.jpg](./example/n02391049_3290_epoch_120.jpg) | ![n02391049_3290_epoch_160.jpg](./example/n02391049_3290_epoch_160.jpg) | ![n02391049_3290_epoch_200.jpg](./example/n02391049_3290_epoch_200.jpg) | 

# 效果不好的案例

| 原图                                                          | 官方                                                          | 复现                                                        |
|-------------------------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------|
| ![n02381460_640_real.jpg](example/n02381460_640_real.jpg)   | ![n02381460_640_fake.jpg](example/n02381460_640_fake.jpg)   | ![n02381460_640_our.jpg](example/n02381460_640_our.jpg)   |
| ![n02391049_3310_real.jpg](example/n02391049_3310_real.jpg) | ![n02391049_3310_fake.jpg](example/n02391049_3310_fake.jpg) | ![n02391049_3310_our.jpg](example/n02391049_3310_our.jpg) |

