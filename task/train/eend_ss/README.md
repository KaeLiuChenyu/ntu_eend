Train EEND-SS
===========================


****

# Folders for train EEND-SS
1. ntu_eend/task/train/eend_ss
   1. config
   2. data
   3. train_scr
   4. utils
   5. run.sh

>i.config
>>config files

>ii.data
>>Includes train/dev/test folders. Check ntu_eend/additional folders for dataset preparation.

>iii.train_scr
>> Scripts needed for training EEND-SS

>iv.utils
>>Utilities

>v.run.py
>>Main script. 


## **Usage**
Step 0(Dataset):
Please check [ntu_eend/additional/prepare_librimix](https://github.com/KaeLiuChenyu/ntu_eend/tree/main/additional/prepare_librimix)
>2&3 speaker model:
>>Use Libri2Mix & Libri3Mix
```
pretrain_stage=false
adapt_stage=true
pretrain_model="path of the pre-trained 2 spk model"
```

>2 speaker model:
>>Use Libri2Mix
```
pretrain_stage=true
adapt_stage=false
```

Step 1:
```
python -m pip install -r requirements.txt
```

Step 2:
```
./run.sh
```



