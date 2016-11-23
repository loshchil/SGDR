# SGDR: Stochastic Gradient Descent with Restarts
Lasagne implementation of SGDR on WRNs from "SGDR: Stochastic Gradient Descent with Restarts" by Ilya Loshchilov and Frank Hutter (http://arxiv.org/abs/1608.03983)  
This code is based on Lasagne Recipes available at
https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
and on WRNs implementation by Florian Muellerklein available at
https://gist.github.com/FlorianMuellerklein/3d9ba175038a3f2e7de3794fa303f1ee

The only input is "iscenario" index used to reproduce the experiments given in the paper   
scenario #1 and #2 correspond to the original multi-step learning rate decay on CIFAR-10  
scenarios [3-6] are 4 options for our SGDR  
scenarios [7-10] are the same options but for 2 times wider WRNs, i.e., WRN-28-20  
scenarios [11-20] are the same as [1-10] but for CIFAR-100  
scenarios [21-28] are the the original multi-step learning rate decay for 2 times wider WRNs on CIFAR-10 and CIFAR-100    

The best reported results in the paper are by SGDR with T0 = 10 and Tmult = 2  
**3.74%** on **CIFAR-10** (median of 2 runs of iscenario #10)  
**18.70%** on **CIFAR-100** (median of 2 runs of iscenario #20) 

Ensembles of WRN-28-10 models trained by SGDR show  
**3.14%** on **CIFAR-10**  
**16.21%** on **CIFAR-100**  
The latest version of the paper is available at 
https://openreview.net/pdf?id=Skq89Scxx



