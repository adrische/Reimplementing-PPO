import numpy as np

class OnlineStats: # TODO replace by observation wrapper
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0 # Sum of squares of differences from the current mean
        self.std = 0.0
        self.variance = 0.0

    def update(self, new_value):
        """Update statistics with a single new value."""
        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean # Delta between the new value and the new mean
        self.M2 += delta * delta2
        self.variance = 0.0 if self.n < 2 else self.M2 / (self.n - 1)
        self.std = np.sqrt(self.variance)



# 8. Reward Scaling, 9. Reward Clipping
# Custom version of NormalizeReward to save the unmodified original reward
# Copied from https://gymnasium.farama.org/_modules/gymnasium/wrappers/stateful_reward/#NormalizeReward
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers.utils import RunningMeanStd

class NormalizeRewardCustom(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        self.return_rms = RunningMeanStd(shape=())
        self.discounted_reward = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon
        self._update_running_mean = True
        self.original_reward = np.array([0.0]) # new

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(action)

        self.original_reward = reward # new

        # Using the `discounted_reward` rather than `reward` makes no sense but for backward compatibility, it is being kept
        self.discounted_reward = self.discounted_reward * self.gamma * (
            1 - terminated
        ) + float(reward)
        if self._update_running_mean:
            self.return_rms.update(self.discounted_reward)

        # We don't (reward - self.return_rms.mean) see https://github.com/openai/baselines/issues/538
        normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return obs, normalized_reward, terminated, truncated, info