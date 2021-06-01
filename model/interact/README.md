# interact

A simple MLP to determine what happens when you apply action A to the situation

## Basic experiments
```bash
ctpu up --tpu-size=v3-8 --tf-version 1.15.3 --noconf --tpu-only

# Train it
nohup python train.py configs/base.yaml > base.txt &

# Then you can eval it
python train.py configs/base.yaml
```