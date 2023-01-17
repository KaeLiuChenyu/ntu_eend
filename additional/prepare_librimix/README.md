Prepare LibriMix
===========================


****

## **Usage**
1.Modify KALDI_ROOT in path.sh:
```
KALDI_ROOT=path/to/your/kaldi
```
2.Link steps and utils folders:
```
ln -sf path/to/your/kaldi/egs/wsj/s5/steps /prepare_librimix
ln -sf path/to/your/kaldi/egs/wsj/s5/utils /prepare_librimix  
```
3.Set number of speaker:
>Train 2 speaker model(Models that can only handle 2 speakers)
>>spk_num=2
>> #The dataset will contain audio from Libri2Mix

>Train 3 speaker model(Models that can handle 2 or 3 speakers)
>>spk_num=3
>> #The dataset will contain audio from Libri2Mix and Libri3Mix(For adapting training)

4.Run:
```
./run.sh
```
5.Result:
Test:
```
cp -r dump/raw/test task/infer/your_task/data
```
Train:
```
cp -r dump/raw/train task/train/your_task/data
cp -r dump/raw/dev task/train/your_task/data
```
