# tfrecord

This folder contains how I turned the data produced by the sampler into `.tfrecord` format. The script you probably want is `to_tfrecord_v2.py`. I ran it like this:

```bash
num_folds_train=256
num_folds_val=128
mkdir -p logs

parallel -j 4 --will-cite "python to_tfrecord_v2.py -no_1ary -no_2ary -fold={1} -num_folds=${num_folds_val} -fns_list val_fns.txt -out_path gs://YOURPATHHERE/val > logs/val{1}.txt" ::: $(seq 0 $((num_folds_val-1)))

parallel -j 4 --will-cite "python to_tfrecord_v2.py -no_1ary -no_2ary -fold={1} -num_folds=${num_folds_train} -fns_list train_fns.txt -out_path gs://YOURPATHHERE/train > logs/train{1}.txt" ::: $(seq 0 $((num_folds_train-1)))

```