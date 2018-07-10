## VGG
### VGG16/VGG19
 * conv1_1: 224x224x3 -> 224x224x64
 * conv1_2: 224x224x64 -> 224x224x64
 * max pool1: 224x224x64 -> 112x112x64
 * conv2_1: 112x112x64 -> 112x112x128
 * conv2_2: 112x112x128 -> 112x112x128
 * max pool2: 112x112x128 -> 56x56x128
 * conv3_1: 56x56x128 -> 56x56x256
 * conv3_2: 56x56x256 -> 56x56x256
 * conv3_3: 56x56x256 -> 56x56x256
 * conv3_4: 56x56x256 -> 56x56x256 <- vgg19有这一层网络
 * max pool3: 56x56x256 -> 28x28x256
 * conv4_1: 28x28x256 -> 28x28x512
 * conv4_2: 28x28x512 -> 28x28x512
 * conv4_3: 28x28x512 -> 28x28x512
 * conv4_4: 28x28x512 -> 28x28x512 <- vgg19有这一层网络
 * max pool4: 28x28x512 -> 14x14x512
 * conv5_1: 14x14x512 -> 14x14x512
 * conv5_2: 14x14x512 -> 14x14x512
 * conv5_3: 14x14x512 -> 14x14x512
 * conv5_4: 14x14x512 -> 14x14x512 <- vgg19有这一层网络
 * max pool5: 7x7x512
 * fc6: 7x7x512 -> 4096
 * fc7: 4096 -> 4096
 * fc8: 4096 -> 1000
 * 不算pool，vgg16一共有16层网络，因此叫做vgg16，vgg19有19层，因此叫做vgg19

### vgg16和vgg19使用原作者的weights进行predict
 * ./python test.py vgg16/vgg19 {weights path} {predict file path}
 * output ->
   (292, 0.74020416, 'tiger, Panthera tigris')
   (282, 0.25786862, 'tiger cat')
   (290, 0.00097516429, 'jaguar, panther, Panthera onca, Felis onca')
   (340, 0.00023462022, 'zebra')
   (288, 0.0001896271, 'leopard, Panthera pardus')
 ### 用vgg网络对自己的数据进行training
   1. 整个网络training：python train.py vgg16/vgg19
   2. 只对最后一层fine turn：python train.py vgg16/vgg19 {any}
   
