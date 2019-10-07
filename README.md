# AVT: Unsupervised Learning of Transformation Equivariant Representations by Autoencoding Variational Transformations
The project website for "Autoencoding Variational Transformations"

## Abstract 
The learning of Transformation-Equivariant Representations (TERs), which is introduced by Hinton et al.,
has been considered as a principle to reveal visual structures under various transformations. It contains the celebrated Convolutional Neural Networks (CNNs) as a special case that only equivary to the translations. In contrast, we seek to train TERs for a generic class of transformations and train them in an unsupervised fashion. To this end, we present a novel principled method by Autoencoding Variational Transformations (AVT), compared with the conventional approach to autoencoding data. Formally, given transformed images, the AVT seeks to train the networks by maximizing the mutual information between the transformations and representations. This ensures the resultant TERs of individual images contain the intrinsic information about their visual structures that would equivary extricably under various transformations in a generalized nonlinear case. Technically, we show that the resultant optimization problem can be efficiently solved by maximizing a variational lower-bound of the mutual information. This variational approach introduces a transformation decoder to approximate the intractable posterior of transformations, resulting in an autoencoding architecture with a pair of the representation encoder and the transformation decoder. Experiments demonstrate the proposed AVT model sets a new record for the performances on unsupervised tasks, greatly closing the performance gap to the supervised models.

## Formulation

| ![AVT](https://github.com/maple-research-lab/AVT-pytorch/blob/master/resources/AVT.PNG) |
|:--:| 
| *Figure 1. The architecture of the proposed AVT. The original and transformed images are fed through the encoder pθ where 1 denotes an identity transformation to generate the representation of the original image. The resultant representations ˜z and z of original and transformed images are sampled and fed into the transformation decoder qφ from which the transformation t is sampled.* |

As illustrated in Figure 1, we implement the transformation decoder by using a Siamese encoder network
with shared weights to represent the original and transformed images with ˜z and z respectively, where the mean and the variance the sampled transformation are predicted from the concatenation of both representations.

For details, please refer to [our paper](https://arxiv.org/pdf/1903.10863.pdf).

## Run our codes
### Requirements
- Python == 2.7
- pytorch == 1.0.1
- torchvision == 0.2.1
- PIL == 5.4.1
### Note
Please use the torchvision with version 0.2.1. The code does not support the newest version of torchvision.
### Cifar10
    cd cifar
Unsupervised learning:

    CUDA_VISIBLE_DEVICES=0 python main.py --cuda --outf ./output --dataroot $YOUR_CIFAR10_PATH$ 
Supervised evaluation with two FC layers:

    python classification.py --dataroot $YOUR_CIFAR10_PATH$ --epochs 200 --schedule 100 150 --gamma 0.1 -c ./output_cls --net ./output/net_epoch_4499.pth --gpu-id 0
### ImageNet 
    cd ImageNet
Generate and save 0.5 million projective transformation parameters:

    python save_homography.py
Unsupervised learning:

    CUDA_VISIBLE_DEVICES=0 python main.py --exp ImageNet_Unsupervised
Supervised evaluation with non-linear classifiers:

    CUDA_VISIBLE_DEVICES=0 python main.py --exp ImageNet_NonLinearClassifiers
Supervised evaluation with linear classifiers (max pooling):

    CUDA_VISIBLE_DEVICES=0 python main.py --exp ImageNet_LinearClassifiers_Maxpooling
Supervised evaluation with linear classifiers (average pooling):

    CUDA_VISIBLE_DEVICES=0 python main.py --exp ImageNet_LinearClassifiers_Avgpooling

To use the pretrained ImageNet model:

    mkdir experiments
    cd experiments
    mkdir ImageNet_Unsupervised

Please download the pre-trained model from the link: https://1drv.ms/u/s!AhnMU9glhsl-x0SBW-EI709dM5pc?e=KZyQg0 and put the models under ./experiments/ImageNet_Unsupervised

### Places205
Firstly pretrain the model on Imagenet, then evalutate the model with linear classifiers (max pooling):

    CUDA_VISIBLE_DEVICES=0 python main.py --exp Places205_LinearClassifiers_Maxpooling
    
Supervised evalutation with linear classifiers (average pooling):

    CUDA_VISIBLE_DEVICES=0 python main.py --exp Places205_LinearClassifiers_Avgpooling

## Citation

Guo-Jun Qi, Liheng Zhang, Chang Wen Chen, Qi Tian. AVT: Unsupervised Learning of Transformation Equivariant Representations by Autoencoding Variational Transformations, in Proceedings of IEEE International Conference on Computer Vision (ICCV), 2019. [[pdf](https://arxiv.org/pdf/1903.10863.pdf)]

## Disclaimer

Some of our codes reuse the github project [FeatureLearningRotNet](https://github.com/gidariss/FeatureLearningRotNet).  

## License

This code is released under the MIT License.
