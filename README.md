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
 * 论文中的网络结构细节处有错误，会导致resnet的block17和block8因为channel不匹配而无法相加，代码实现中做了修改。
 * goole tensorflow中的网络实现和论文中差别很大，本文的实现还是尽量按照论文中的网络结构[reference code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py)
### 网络训练
      ```sh
     $ python train.py ./cfg/inception-resnet-v2.json
     ```
## Resnet V2
### block结构
 * block结构参考论文:[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
 
 论文主要就是尝试哪种block结构能够提升网络性能。经过多种尝试和数据分析后认为block结构从a改进为b会提升网络效果。
 ![orginal residual unit -> proposed residual unit](https://github.com/stesha2016/tensorflow-classification/blob/master/image/residual_unit.png)
 * 代码实现参考了[reference code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py)
 
 google的代码实现比较炫技,不过确实扩展网络非常容易,从resnet_50到resnet_200只需要调整几个参数就可以实现.
 
 另外解释一下50层,101层等的组成
 50层: (3+4+6+3)*3 + 2 = 50 每个block有3个conv层,还要加上block开始之前的1层和block结束后的1层
> blocks = [
 	resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),	
 	resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),	
	resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),	
	resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)
 ]

 101层: (3+4+23+3)*3 + 2 = 101
> blocks = [
	resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
	resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
	resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
	resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)
]

 152层: (3+8+36+3)*3 + 2 = 152
> blocks = [
	resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
	resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
	resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
	resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)
]

 200层: (3+24+36+3)*3 + 2 = 200
> blocks = [
	resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
	resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
	resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
	resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)
]
### 网络训练
      ```sh
     $ python train.py ./cfg/resnet-v2.json
     ```
