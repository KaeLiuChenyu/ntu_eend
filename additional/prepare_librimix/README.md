Prepare LibriMix
===========================


****

## **Usage**
1.Modify kaldi path:
```
Modify KALDI_ROOT in path.sh
KALDI_ROOT=path/to/your/kaldi
```
2.Link steps and utils folders:
```
ln -sf path/to/your/kaldi/egs/wsj/s5/steps /prepare_librimix
!ln -sf path/to/your/kaldi/egs/wsj/s5/utils /prepare_librimix  
```
3.Run:
```
./run.sh
```
4.Run:
```
The result will be saved in dump/raw/train&dev&test
```
