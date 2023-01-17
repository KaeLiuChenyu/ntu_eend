Infer EEND-SS
===========================


****

# Folders for Infer eend_ss

1. ntu_eend/task/infer/eend_ss
   1. infer_scr
   2. run.py


>i.infer_scr
>> Scripts needed for infering eend_ss

>v.run.py
>>Main script. 


## **Usage**
Step 0(Dataset):
Please check [ntu_eend/additional/prepare_librimix](https://github.com/KaeLiuChenyu/ntu_eend/tree/main/additional/prepare_librimix)

Step 1:
```
Modify 
train_config=/exp/train/config.yaml # automatically generated files in training exp folder
model_file=/valid.si_snr_loss.best_old.pth # path to checkpoint file
threshold=
in run.sh
```
Step 2:
```
./run.sh
```
Step 3:
generate_rttm=true:
```
The final result will be saved in result/pre.rttm
```
generate_rttm=false:
```
The prediction of test audio will be saved in result/predictions as .npy files
```