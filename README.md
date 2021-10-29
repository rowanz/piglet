# piglet
PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World [ACL 2021]
This repo contains code and data for PIGLeT. If you like this paper, please cite us:
```
@inproceedings{zellers2021piglet,
    title={PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World},
    author={Zellers, Rowan and Holtzman, Ari and Peters, Matthew and Mottaghi, Roozbeh and Kembhavi, Aniruddha and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
    year={2021}
}
```

See more at [https://rowanzellers.com/piglet](https://rowanzellers.com/piglet)


# What this repo contains

## Physical dynamics model
* You can get data yourself by sampling trajectories in [sampler/](sampler/) and then converting them to `tfrecord` (which is the format I used) in [tfrecord/](tfrecord). I also have the exact tfrecords I used at `gs://piglet-data/physical-interaction-tfrecords/` -- they're big files so I turned on 'requester pays' for them.
    * These TFrecords contain the world state that we trained on for this paper, and additionally, image frames 😄.
    * You can download the tfrecords using something like `gsutil -u $myusername gs://piglet/physical-interaction-tfrecords/*.tfrecord .`
    * Confusingly, the training files are numbered like `0XXXof0256`, but there is no fold 0018. I only realized this after making these tfrecords 😅 so in other words, this was used for my internal experiments too. Basically you hopefully shouldn't have to worry about this at all, just don't panic if you don't see a file like `train-0018of0256.tfrecord`!


* You can pretrain the model and evaluate it in [model/interact/train.py](model/interact/train.py) and [model/interact/intrinsic_eval.py](model/interact/intrinsic_eval.py)
* Alteratively feel free to use my checkpoint: `gs://piglet/checkpoints/physical_dynamics_model/model.ckpt-5420`

## Language model
* You can process data (also in tfrecord format) using [data/zeroshot_lm_setup/prepare_zslm_tfrecord.py](data/zeroshot_lm_setup/prepare_zslm_tfrecord.py), or download at `gs://piglet-data/text-data/`. I have both 'zero-shot' tfrecord data, basically a version of BookCorpus and Wikipedia where certain concepts are filtered out, as well as non-zero shot (regularly processed). This was used to evaluate generalization to new concepts.
* Train the model using [model/lm/train.py](model/lm/train.py)
* Alternatively, feel free to just use my checkpoint: `gs://piglet/checkpoints/language_model/model.ckpt-20000`

## Tying it all together
* Everything you need for this is in [model/predict_statechange/](model/predict_statechange/) building on both the physical dynamics model and language model pretrained.
* I have annotations in [data/annotations.jsonl](data/annotations.jsonl) for training and evaluating both tasks -- PIGPeN-NLU and PIGPeN-NLG. 
* Alternatively you can download my checkpoints at `gs://piglet/checkpoints/pigpen-nlu-model/` for NLU (predicting state change given english text) or `gs://piglet/checkpoints/pigpen-nlg-model/` for NLG.


That's it! 

## Getting the environment set up
I used TPUs for this project so those are the only things I support right now, sorry!

I used tensorflow 1.15.5 and TPUs for this project. My recommendation is to use `ctpu` to start up a VM with access to a `v3-8` TPU. Then, use the following command to install dependencies:
```
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p ~/conda && \
     rm ~/miniconda.sh && \
     ~/conda/bin/conda install -y python=3.7 tqdm numpy pyyaml scipy ipython mkl mkl-include cython typing h5py pandas && ~/conda/bin/conda clean -ya
     
echo 'export PATH=~/conda/bin:$PATH' >>~/.bashrc
source ~/.bashrc
pip install "tensorflow==1.15.5"
pip install --upgrade google-api-python-client oauth2client
pip install -r requirements.txt
```
