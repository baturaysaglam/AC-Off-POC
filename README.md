# Off-Policy Correction for Actor-Critic Algorithms in Deep Reinforcement Learning
PyTorch implementation of the _Actor-Critic Off-Policy Correction_ algorithm (AC-Off-POC). If you use our code or data please cite the [paper](https://arxiv.org/abs/2208.00755).

Note that the implementation of the [DDPG](https://arxiv.org/abs/1509.02971) and [TD3](https://arxiv.org/abs/1802.09477) algorithms are heavily based on the [author's Pytorch implementation of the TD3 algorithm](https://github.com/sfujim/TD3). 

The algorithm is tested on [MuJoCo](https://gym.openai.com/envs/#mujoco) and [Box2D](https://gym.openai.com/envs/#box2d) continuous control tasks.

### Results
Learning curves are found under [./Learning Curves](https://github.com/baturaysaglam/AC-Off-POC/tree/main/Learning%20Curves).. Corresponding learning figures are found under [./Learning Figures](https://github.com/baturaysaglam/AC-Off-POC/tree/main/Learning%20Figures). Each learning curve is formatted as NumPy arrays of 1001 evaluations (1001,). Each evaluation corresponds to the average reward from running the policy for 10 episodes without exploration and updates. The randomly initialized policy network produces the first evaluation. Evaluations are performed every 1000 time steps, over 1 million time steps for 10 random seeds.

### Computing Infrastructure
Following computing infrastructure is used to produce the results.
| Hardware/Software  | Model/Version |
| ------------- | ------------- |
| Operating System  | Ubuntu 18.04.5 LTS  |
| CPU  | AMD Ryzen 7 3700X 8-Core Processor |
| GPU  | Nvidia GeForce RTX 2070 SUPER |
| CUDA  | 11.1  |
| Python  | 3.8.5 |
| PyTorch  | 1.8.1 |
| OpenAI Gym  | 0.17.3 |
| MuJoCo  | 1.50 |
| Box2D  | 2.3.10 |
| NumPy  | 1.19.4 |

### Usage - DDPG & TD3
```
usage: main.py [-h] [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--start_time_steps N] [--buffer_size BUFFER_SIZE]
               [--eval_freq N] [--max_time_steps N] [--exploration_noise G]
               [--batch_size N] [--kl_div_var KL_DIV_VAR] [--discount G]
               [--tau G] [--policy_noise G] [--noise_clip G] [--policy_freq N]
               [--save_model] [--load_model LOAD_MODEL]
```

### Arguments - DDPG & TD3
```
optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: AC-Off-POC_DDPG)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --start_time_steps N  Number of exploration time steps sampling random actions (default: 1000)
  --buffer_size BUFFER_SIZE Size of the experience replay buffer (default: 1000000)
  --eval_freq N         Evaluation period in number of time steps (default: 1000)
  --max_time_steps N    Maximum number of steps (default: 1000000)
  --exploration_noise G Std of Gaussian exploration noise
  --batch_size N        Batch size (default: 256)
  --kl_div_var KL_DIV_VAR Diagonal entries of the reference Gaussian for the Deterministic SAC
  --discount G          Discount factor for reward (default: 0.99)
  --tau G               Learning rate in soft/hard updates of the target networks (default: 0.005)
  --policy_noise G      Noise added to target policy during critic update
  --noise_clip G        Range to clip target policy noise
  --policy_freq N       Frequency of delayed policy updates
  --save_model          Save model and optimizer parameters
  --load_model LOAD_MODEL Model load file name; if empty, does not load
  ```
  
### Usage - SAC
```
usage: main.py [-h] [--policy POLICY] [--policy_type POLICY_TYPE] [--env ENV]
               [--seed SEED] [--gpu GPU] [--start_steps N]
               [--off_poc_update_start_steps N] [--buffer_size BUFFER_SIZE]
               [--eval_freq N] [--num_steps N] [--batch_size N]
               [--kl_div_var KL_DIV_VAR] [--hard_update G]
               [--updates_per_step N] [--target_update_interval N] [--alpha G]
               [--automatic_entropy_tuning G] [--reward_scale N] [--gamma G]
               [--tau G] [--lr G] [--hidden_size N]
```

### Arguments - SAC
```
optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: AC-Off-POC_SAC)
  --policy_type POLICY_TYPE Policy Type: Gaussian | Deterministic (default: Gaussian)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --start_steps N       Number of exploration time steps sampling random actions (default: 1000)
  --off_poc_update_start_steps N Multiple of exploration time steps to start AC-Off-POC updates (default: 50)
  --buffer_size BUFFER_SIZE Size of the experience replay buffer (default: 1000000)
  --eval_freq N         evaluation period in number of time steps (default: 1000)
  --num_steps N         Maximum number of steps (default: 1000000)
  --batch_size N        Batch size (default: 256)
  --kl_div_var KL_DIV_VAR Diagonal entries of the reference Gaussian for the Deterministic SAC
  --hard_update G       Hard update the target networks (default: True)
  --updates_per_step N  Model updates per training time step (default: 1)
  --target_update_interval N Number of critic function updates per training time step (default: 1)
  --alpha G             Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)
  --automatic_entropy_tuning G Automatically adjust α (default: False)
  --reward_scale N      Scale of the environment rewards (default: 5)
  --gamma G             Discount factor for reward (default: 0.99)
  --tau G               Learning rate in soft/hard updates of the target networks (default: 0.005)
  --lr G                Learning rate (default: 0.0003)
  --hidden_size N       Hidden unit size in neural networks (default: 256)

  ```

### Bibtex
```
@misc{https://doi.org/10.48550/arxiv.2208.00755,
  doi = {10.48550/ARXIV.2208.00755},
  
  url = {https://arxiv.org/abs/2208.00755},
  
  author = {Saglam, Baturay and Cicek, Dogan C. and Mutlu, Furkan B. and Kozat, Suleyman S.},
  
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Off-Policy Correction for Actor-Critic Algorithms in Deep Reinforcement Learning},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
