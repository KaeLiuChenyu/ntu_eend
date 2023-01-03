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
First:
```
pip install jaconv

pip install jamo

pip install torch_complex

pip install espnet_tts_frontend

pip install fast_bss_eval

pip install ci_sdr
```

Second:
```
./run.sh
```

