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
Check ntu_diar/task/Train/x_vector

## **Inference**
Check ntu_diar/task/Inference/silero_xvector
