# Reinforcement Learning: Zero to PPO

This is my attempt at step-by-step reimplementing the paper [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347), using only resources available at the time of publication (2017). I didn't have any previous knowledge in reinforcement learning, so please take what is written here with a grain of salt.

> **Draft version:** I'm intentionally making this available at an early stage to get (human) feedback. Please don't hesitate to reach out.



## Outline

The outline is as follows:

Status: open for feedback:

1. **REINFORCE:** The first policy gradient algorithm presented by [Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html) applied to the CartPole environment. [Notebook](notebooks/REINFORCE%20CartPole.ipynb)

2. **REINFORCE with baseline:** Introduction of the baseline, naive hyper-parameter search, the PPO policy network structure. [Notebook](notebooks/REINFORCE%20with%20baseline%20LunarLander.ipynb)

* _Interlude:_ Engineering improvements to REINFORCE with baseline. A few items that are used even though they are not part of the algorithm description. [Notebook](notebooks/REINFORCE%20with%20baseline%20LunarLander%20engineering%20improvements.ipynb)

* _Interlude:_ Continuous action space. A description of how to model continuous action distributions with a normal distribution, how to do back-propagation, and how to treat bounded action spaces. [Notebook](notebooks/Interlude%20continuous%20action%20space.ipynb)

3. **REINFORCE with baseline with continuous action space.** Preparation for the Mujoco environments used in the PPO paper. Establishing reference performance against which PPO to compare to. [Notebook](notebooks/REINFORCE%20with%20baseline%20continuous.ipynb)


Status: first draft:

1. **PPO for Mujoco environments.** Collection of a batch of several episodes, gradient descent using samples of time-steps, for several epochs. Introduction of the actor-critic approach with generalized advantage estimation. Correction for importance sampling, and clipped objective. [Notebook](notebooks/PPO%20MuJoCo.ipynb)


Status: Planned:

* _Interlude:_ Further study the effects of generalized advantage estimation, importance sampling, and clipped objective.

* _Interlude:_ Performance improvements.

5. PPO for Roboschool. Parallel episode roll-outs, adaptive learning rate.

6. PPO for Atari. Image preprocessing, parameter sharing between policy and value function, entropy bonus.

7. Comparison with reference implementations.



## Learning goals

The goal is to better understand reference implementations of PPO, such as 

* https://github.com/XinJingHao/PPO-Continuous-Pytorch/
* https://github.com/vwxyzjn/cleanrl
* https://stable-baselines3.readthedocs.io/en/master/

Performance or best possible abstraction was not a goal. The focus was on easy-to-follow code and step-by-step introduction and analysis of additional algorithm components.




## References

### 2017 and earlier

Paper https://arxiv.org/pdf/1707.06347

2017 Berkeley Deep RL Bootcamp 
- https://sites.google.com/view/deep-rl-bootcamp/home
- Labs https://sites.google.com/view/deep-rl-bootcamp/labs
- Lecture 4B Policy Gradients Revisited https://www.youtube.com/watch?v=tqrcjHuNdmQ
- Lecture 5: Natural Policy Gradients, TRPO, PPO https://www.youtube.com/watch?v=xvRrgxcpaHY
- Lecture 6: Nuts and Bolts of Deep RL Experimentation https://www.youtube.com/watch?v=8EcdaCk9KaQ
- Pong from pixels: 
  - Blog https://karpathy.github.io/2016/05/31/rl/
  - Code https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

Old official PPO implementation
https://github.com/openai/baselines

2016 NIPS tutorial
- https://nips.cc/virtual/2016/tutorial/6198
- https://media.nips.cc/Conferences/2016/Slides/6198-Slides.pdf

John Schulman's thesis from 2016 https://escholarship.org/content/qt9z908523/qt9z908523.pdf

4 more John Schulman lectures https://www.youtube.com/watch?v=aUrX-rP_ss4

David Silver 2016 Introduction to Reinforcement Learning course
https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ

https://www.coursera.org/specializations/reinforcement-learning


### Post-2017

https://huggingface.co/learn/deep-rl-course/en/unit0/introduction

Lessons learned from reimplementing reinforcement learning papers:
- https://amid.fish/reproducing-deep-rl
- https://github.com/mrahtz/ocd-a3c?tab=readme-ov-file#ocd-a3c
- https://www.agentydragon.com/posts/2021-10-18-rai-ml-mistakes-3.html

More courses
- https://rail.eecs.berkeley.edu/deeprlcourse/
- https://github.com/yandexdataschool/Practical_RL

The 37 Implementation Details of PPO (2022) https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

