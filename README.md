# SENetV2-Aggregated-dense-layer-for-channelwise-and-global-representations
This is the official implementation of our paper "SENetV2 Aggregated dense layer for channelwise and global representations". In this paper, we propose a novel multi dense layer for squeeze and excitation network. Multi dense layer module is similar to multiconv aggregated layer as in Inception module.


# Abstract

Convolutional Neural Networks (CNNs) have revolutionized image classification by extracting spatial features and enabling state-of-the-art accuracy in vision-based tasks. The squeeze and excitation network proposed module gathers channelwise representations of the input. Multilayer perceptrons (MLP) learn global representation from the data and in most image classification models used to learn extracted features of the image. In this paper, we introduce a novel aggregated multilayer perceptron, a multi-branch dense layer, within the Squeeze excitation residual module designed to surpass the performance of existing architectures. Our approach leverages a combination of squeeze excitation network module with dense layers. This fusion enhances the network's ability to capture channel-wise patterns and have global knowledge, leading to a better feature representation. This proposed model has a negligible increase in parameters when compared to SENet. We conduct extensive experiments on benchmark datasets to validate the model and compare them with established architectures. Experimental results demonstrate a remarkable increase in the classification accuracy of the proposed model.

# SENetV2 module

![SENetV2 module](https://github.com/mahendran-narayanan/SENetV2-Aggregated-dense-layer-for-channelwise-and-global-representations/blob/main/data/senetv2.png)

# Results

## CIFAR-10

![CIFAR-10](https://github.com/mahendran-narayanan/SENetV2-Aggregated-dense-layer-for-channelwise-and-global-representations/blob/main/data/cifar10.png)

## CIFAR-100

![CIFAR-100](https://github.com/mahendran-narayanan/SENetV2-Aggregated-dense-layer-for-channelwise-and-global-representations/blob/main/data/cifar100.png)

Link to the paper [SENetV2](https://arxiv.org/abs/2311.10807)
