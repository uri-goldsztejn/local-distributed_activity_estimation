# Estimating uterine activity from electrohysterogram measurements via statistical tensor decomposition

This is the code developed for our paper "[Estimating uterine activity from electrohysterogram measurements
via statistical tensor decomposition](https://www.sciencedirect.com/science/article/abs/pii/S1746809423003324?casa_token=ay36NKGerGIAAAAA:G2l5ZgFyFZddwOfo0o5cyQNSABfMcNJePfbjA9Mben30MwLG77NOLfhaAxyggK_n3OzJWBt0)"  

A preprint of our paper can be found [here](https://arxiv.org/abs/2209.02183).   

Uri Goldsztejn, Arye Nehorai  
Washington University in St. Louis, 2023

*If you find this code useful, please consider citing.*

## Content
* Overview
* Files
* Requirements
* Citation
* Contact

## Overview

Complications during pregnancy and labor are common and can be especially detrimental in populations with limited access to healthcare. Electrohysterogram (EHG) measurements noninvasively record uterine electrical activity and are a promising technology for detecting and predicting uterine electrical abnormalities. However, EHGs record electrical activity from various sources besides the uterus, e.g., maternal respiration and electrical activity from the abdominal muscles.
To address this limitation, we developed a statistical tensor decomposition to separate localized activity corresponding to uterine activity from temporally and spatially distributed electrical activity corresponding to more uniformly distributed physiological processes, such as maternal respiration.
Using simulated EHG measurements, we showed that our method can estimate localized activity more accurately than existing matrix and tensor methods. We also showed that our method can denoise uterine contractions with higher SNRs, defined as power ratios between EHG bursts and segments with only baseline EHG activity, than alternative methods.


## Files

* *tensor_decomposition* - Implements the tensor decomposition described in our work.
* *utils/* - auxiliary functions.
* *example/* - data and usage examles.

## Software requirements

This tensor decomposition requires the following submodules:
* The Tensor toolbox, which can be downloaded from: [tensortoolbox.org](https://www.tensortoolbox.org/)
* ckronx: An efficient computation of Kronecker products (https://www.mathworks.com/matlabcentral/fileexchange/74107-ckronx-efficient-computation-with-kronecker-products), MATLAB Central File Exchange.


## Reference

Goldsztejn, U., & Nehorai, A. (2023). Estimating uterine activity from electrohysterogram measurements via statistical tensor decomposition. Biomedical Signal Processing and Control, 85, 104899.
