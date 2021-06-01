# sampler

This is code for sampling trajectories used to train the Physical Dynamics Model for PIGLeT.

The idea is that we want to sample trajectories that might be 'interesting' for learning stuff from. If you move randomly in AI2-THOR you probably won't do anything interesting, you'll just bump around and get stuck in a corner or something. The focus here is being as stochastic as possible while still having high coverage of interesting situations.

## Running the code
* First I had to run a few things for linking burners and knobs, so the planner knows which burners it should try to turn on, but you don't need to because I've included the data already! The scripts `_compute_size_multipliers.py` and `_link_burners_and_knobs.py` are there just in case.
* Now you can run `sample_trajectories.py`. In fact you can do something like this to run it on several GPUs (here my GPUs are `0`,`1` and `2`)
```bash
mkdir -p logs
parallel --link -j 12 --will-cite "python sample_trajectories.py -out_path=/home/rowan/datasets3/piglet/ -seed={1} -display={2} > logs/{1}.txt" ::: $(seq 16000 17000) ::: 0 1 2
```
* You can visualize with `visualize_samples.py`.
