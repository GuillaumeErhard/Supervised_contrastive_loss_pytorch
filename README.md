<h1 align="center">Supervised Contrastive Loss Pytorch</h1>

<div align="center">
  <img width="50%" alt="SimCLR Illustration" src="https://raw.githubusercontent.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/main/images/representation_explained.gif">
</div>

This is an independent reimplementation of the [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) paper.   
Go [here](https://github.com/HobbitLong/SupContrast) if you want to go to an implementation from one the author in torch
and [here](https://github.com/google-research/google-research/tree/master/supcon) for the official in tensorflow. 
The goal of this repository is to provide a straight to the point implementation and experiment to answer 
specific question.

# Results

## Accuracy (%) on CIFAR-10 


|Architecture |Cross-entropy| Cross-entropy + Auto-augment|SupContrast + Auto-augment|
|-------|---|---|---|
|  ResNet20 | 91.25 % [*](https://arxiv.org/abs/1512.03385)|  92.71 % | 93.51 %


## Accuracy (%) on CIFAR-100 
|Architecture |Cross-entropy| Cross-entropy + Auto-augment|SupContrast + Auto-augment|
|-------|---|---|---|
|  ResNet20 | 66.05|  66.28 | 68.42 %

# How to use it

## Installation

After creating a virtual environment with your software of choice just run 
```
pip install -r requirements.txt
```

# Usage

A simple run of the following command will give you available script option. Default values will help your replicate 
my results
```
python train.py -h
```



## Some insights gathered

### Some claims from the paper: 

**Is a contrastive epoch taking 50 % more time than a cross-entropy one ?**  
Yes this claim seems inline with mine and official implementation

**Is the use of heavy data-augmentation necessary ?**  
Seems like it. A run without hyper parameter tuning and without AutoAugment but with the same data-augmentation as the original 
ResNet paper yielded a 5 % drop in accuracy compared to the cross-entropy. Although in the paper other data augmentation policies are close 
behind it  contrastive approaches seem to not need sophisticated data augmentation strategies. See original SimCLR [paper](https://arxiv.org/abs/2002.05709)

**Do you need few epochs to train the decoder on the embedding ?**  
Yes definitely. Only 1-2 epochs of cross-entropy on the embedding gave a model close to the best accuracy. Better 
configuration were found after tens of epochs but it was usually only better in the 1e-1 accuracy range.


### Some findings and personal notes

**What is the number of contrastive epoch needed ?**  
The number of epochs necessary to have a good embedding after the contrastive step is higher than a regular cross-entropy.
I did 400/500 epochs while in the official-github the default value is at 1000 epochs and in the paper 700 epochs is mentioned for
ILSVRC-2012. For my test with cross entropy it was at most 700 epochs.

**Why the loss never reaches zero ?**  
The supervised contrastive loss defined in the paper will converge to a constant value, which is batch size dependant.  
The loss as it is described in the paper is analogous to the [Tammes problem](https://en.wikipedia.org/wiki/Tammes_problem) 
where each clusters where projections of a particular class land repel other clusters. Although it is unsolved for such 
high dimension of 128, an approximate solution over dataset statistics can be easily calculated. This could be computationally
 intensive when taking in random configurations at each batch but could be avoided with a sampler given back the same 
 labels configuration. I suspect it might be an easy avenue to reduce the number of epochs needed before convergence. 

**Will this work for very small network ?**  
This approach seems to work also on small network and is one of the addition of this repo. As you can see ResNet-20 results above where
 this approach was better than cross entropy and the model is only
 .3 M parameters. Which is drastically lower than the 20 + M for ResNet-50 on ILSVRC-2012 and the official github.

**Would I recommend using this approach for your specific task ? And will I use it ?**  
One thing that I do like and is the main selling point of this technique is exchanging the boring process of hyper 
parameter tuning for computation. All result presented here only needed one training attempt. 
You just need to decrease the learning rate along the way, whereas with cross-entropy I had to rerun the experiment 
on average 3 times with different learning rate strategy to get the best result shown.  
The other thing that seems to emerge from this paper is that it seems that this method is one of the best in a tabula rasa 
approach. But you can look also in [GradAug](https://arxiv.org/abs/2006.07989), [CutMix](https://arxiv.org/abs/1905.04899) or [Bag of tricks](https://arxiv.org/abs/1812.01187). So it might be a great fit when you are dealing with a problem with non standard images i.e no ILSRVC-2012
like dataset available to pretrained on and it is difficult to collect a ton of unlabelled data also. In the case where you can 
gather a lot of unlabelled data you might have better result with semi-supervised approach like 
[SimCLRv2](https://arxiv.org/abs/2006.10029) or [BYOL](https://arxiv.org/abs/2006.07733). But I guess if 
you are here you know about them.
