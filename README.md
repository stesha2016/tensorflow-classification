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
## MobileNet V1
### 论文解析
* [论文地址](https://arxiv.org/abs/1704.04861)
* 论文中跟网络结构相关的主要有三个部分，其他的内容都是一些网络性能比较
#### 1.引入了一种depthwise convolution的网络结构，将原本的convolution变成了两个结构。一个是depthwise convolution(b),一个是1x1 pointwise convolution(c)
 ![dw block](https://github.com/stesha2016/tensorflow-classification/blob/master/image/mobilenetv1-1.png)
 * 假设input数据为DFxDFxM，kernel为DkxDk,output为DGxDGxN
 * 因此原始的计算量为DKxDKxMxDFxDFxN,改进后的计算量为DkxDKxDFxDFxM+DFxDFxMxN
 ![计算量比例](https://github.com/stesha2016/tensorflow-classification/blob/master/image/mobilenetv1-2.png)
 如果DK为3，则可以缩小接近8／9的计算量
#### 2.网络结构
 ![block structure](https://github.com/stesha2016/tensorflow-classification/blob/master/image/mobilenetv1-3.png)
 ![network structure](https://github.com/stesha2016/tensorflow-classification/blob/master/image/mobilenetv1-4.png)
#### 3.提出了两种参数width multiplier和resolution multiplier
 * width multiplier取值范围0到1之间，用做缩小input channel和output channel -> width multiplier * M, width multiplier * N
 * resolution mulitplier取值范围在0到1之间，用来做小input的size -> resolution multiplier * DF
### 代码实现
 同样参考google代码的实现，用一个_CONV_DEFS展示了网络结构。但是与google的代码稍微有一点不一样，最后一行的DepthSepConv按照论文来实现应该是2，但是google的代码1，不知道是不是改进，不过这里还是按照论文来实现。
### 网络训练 
      ```sh
     $ python train.py ./cfg/mobilenet_v1.json
     ``` 
### MobileNet V2 
#### 论文解析
 * [论文地址](https://arxiv.org/abs/1801.04381)
 * 论文是针对mobilenet v1进行精确度提高的改进。因为depthwise convolution是不能改变channel的，如果上一层的channel比较小，那在DW层的运算就是基于比较小的channel进行，导致学习的特征不多。因此作者的改进思路就是先进行channel升维，然后用DW进行传递，接着对channel进行降维。并且因为最后一层只是要做降维，所以进行linear的activation。
 * MobileNetV1: DW -> PW
 * MobileNetV2: PW -> DW -> PW(Linear)
 ![网络结构](https://github.com/stesha2016/tensorflow-classification/blob/master/image/mobilenetv2-01.png)
 有两种结构，一种是stride为1时，block和residual block类似，另一种是stride为2时的block
 * 计算量计算
 > 从HxWxD经过计算成为HxWxD'有如下三个步骤
 
 > HxWxD ----1x1 PW Relu6----> HxWx(tK)
 
 > HxWx(tK) ----KxK DW Relu6----> HxWx(tK)
 
 > HxWx(tK) ----1x1 PW Linear----> HxWxD'
 
 > 计算量：H*W*D*(tK) + K*K*H*W*(tK) + H*W*(tK)*D' = H*W*(tK)*(D + K^2 + D')
 * 网络结构
 [网络结构](https://github.com/stesha2016/tensorflow-classification/blob/master/image/mobilenetv2-02.png)
 s为2的第一层为2，后面的repeat s为1
 * 从google实现的源码中可以看出，对于channel基本上是除以8后，分成不同的block进行计算的。比如112x112x16会分割成112x112x8和112x112x8两个block进行计算
