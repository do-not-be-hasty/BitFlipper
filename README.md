# BitFlipper
BitFlipper environment in OpenAI gym format


## Problem Statement
Let's say there is a n size binary array (e.g [1,0,0,0,1,1]) and we want to get different array (e.g [0,0,0,1,1,0]) of the same size. Only actions actions allowed are single bit flip at i th position and only get a reward of 0 if the final state is achieved. Otherwise you get a reward -1.

initial=[1,0,0,0,1,1]<br>
 flip at index 2 =[1,0,1,0,1,1]<br>
reward=-1

 flip at index 5=[1,0,1,0,1,0]<br>
 reward=-1 
 
 The goal is to make a agent learn to achieve final state given a initial state.The only observation the agent get are the reward and current state.
 
## Steps to run
Clone the repo<br>
In a conda env / virtualenv :<br> `pip install -e .`
<br>
To run DQN on BitFlipper environment call main() from dqn.py

To run DQN+HER  on BitFlipper environment call main() from dqn_her.py. 
(For this install the repository dqn_her from here https://github.com/sandipan1/dqn_her)

Note that DQN is trained for single state-goal pair while DQN+HER is trained for multiple state-goal pairs.See the Readme in drive link at the bottom for more details.
### Related papers:
Deep Q Networks :http://www.davidqiu.com:8888/research/nature14236.pdf 

Hindsight Experience Replay:https://arxiv.org/pdf/1707.01495.pdf


## Observations
We found that DQN fails even for single state-goal pair as n goes above 15 but DQN+HER could solve problems till n=35.
We tried n=2,5,10,15,20,25,30,35,40. DQN_HER could not solve n=40. Maybe that would require better tuning of hyperparameters.

Here are the results and trained models from our experiments : https://drive.google.com/drive/folders/1yn3Zz-SFtUpw08TmjTVAlQyb4z_vWMf1?usp=sharing

If you want to know more about this, check out our blog on https://deeprobotics.wordpress.com/
