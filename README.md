## VGG
### VGG16/VGG19
 > conv1_1: 224x224x3 -> 224x224x64 @@ conv1_2: 224x224x64 -> 224x224x64 @@ max pool1: 224x224x64 -> 112x112x64
 > conv2_1: 112x112x64 -> 112x112x128 @@ conv2_2: 112x112x128 -> 112x112x128 @@ max pool2: 112x112x128 -> 56x56x128
 > conv3_1: 56x56x128 -> 56x56x256 @@ conv3_2: 56x56x256 -> 56x56x256 @@ conv3_3: 56x56x256 -> 56x56x256 @@ conv3_4: 56x56x256 -> 56x56x256(vgg19有这一层网络) @@ max pool3: 56x56x256 -> 28x28x256
 > conv4_1: 28x28x256 -> 28x28x512 @@ conv4_2: 28x28x512 -> 28x28x512 @@ conv4_3: 28x28x512 -> 28x28x512 @@ conv4_4: 28x28x512 -> 28x28x512(vgg19有这一层网络) @@ max pool4: 28x28x512 -> 14x14x512
 > conv5_1: 14x14x512 -> 14x14x512 @@ conv5_2: 14x14x512 -> 14x14x512 @@ conv5_3: 14x14x512 -> 14x14x512 @@ conv5_4: 14x14x512 -> 14x14x512(vgg19有这一层网络) @@ max pool5: 7x7x512
 > fc6: 7x7x512 -> 4096
 > fc7: 4096 -> 4096
 > fc8: 4096 -> 1000
 * 不算pool，vgg16一共有16层网络，因此叫做vgg16，vgg19有19层，因此叫做vgg19
### vgg16和vgg19使用原作者的weights进行predict
```sh
$ ./python test.py vgg16/vgg19 {weights path} {predict file path}
$ ...
$ (292, 0.74020416, 'tiger, Panthera tigris')
$ (282, 0.25786862, 'tiger cat')
$ (290, 0.00097516429, 'jaguar, panther, Panthera onca, Felis onca')
$ (340, 0.00023462022, 'zebra')
$ (288, 0.0001896271, 'leopard, Panthera pardus')
```
 ### 用vgg网络对自己的数据进行training
 1. 对整个网络training，修改./cfg/vgg.json中的fineturn为false
     ```sh
     $ python train.py ./cfg/vgg.json
     ```
 2. 只对最后一层fine turn, 修改./cfg/vgg.json中的fineturn为true
     ```sh
     $ python train.py ./cfg/vgg.json
     ```

## Inception V4
### 网络结构
 * 网络结构参考论文，非常清晰，[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
 * 代码实现参考google tensorflow中的实现，[reference code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py)
### 网络训练
      ```sh
     $ python train.py ./cfg/inception-v4.json
     ```
## Inception Resnet V2
### 网络结构
 * 网络结构参考论文:[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
 * 论文中的网络有些细节有点错误，会导致resnet的block17和block8因为channel不匹配而无法相加，channel的数量可以根据实际情况来调整。
 * 代码实现部分参考google tensorflow中的实现，但是tensorflow中的网络实现其实和论文中差别很大，本文的实现还是尽量按照论文中的网络结构[reference code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py)
### 网络训练
      ```sh
     $ python train.py ./cfg/inception-resnet-v2.json
     ```
## Resnet V2
### block结构
 * block结构参考论文:[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
 论文主要就是尝试那种block能够提升网络效果。经过多中尝试和数据分析后认为block结构从a改进为b会提升网络效果。
 [original resdual unit]()
 [full pre-activation unit]()
