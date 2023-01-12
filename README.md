NTU EEND
===========================


****

# Folders for ntu_eend
## **scr**
Main script:
1. scr
   1. dataio
   2. nnet
   3. task
   4. utils

>i.dataio
>>Includes dataset loading processing scripts for creating Datasets.

>ii.nnet
>>The components that make up EEND system. This includes the transformer network structure contained in EEND, the RNN structure of EDA, and Speech Separation Network Conv-TasNet.

>iii.task
>>Includes scripts to implement training tasks.

>iv.utils
>>Utilities



## **task**
Training & Inference task:
1. task
   1. Inference
   2. Train


>i.Inference
>>Includes several Inference scripts for EEND models.

>ii.Train
>>Includes several Training scripts for EEND models.


## **additional**
The attached scripts, such as the preparation of datasets:


# Usage

## **Train**
Please check [ntu_diar/task/train/eend_ss](https://github.com/KaeLiuChenyu/ntu_eend/tree/main/task/train/eend_ss)

## **Inference**
Please check [ntu_diar/task/infer/eend_ss](https://github.com/KaeLiuChenyu/ntu_eend/tree/main/task/infer/eend_ss)
