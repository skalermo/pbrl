from abc import abstractmethod
from typing import Tuple, Any

import gymnasium
import numpy as np


def reset_after_done(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs = env.reset()[0]
    return obs, reward, terminated, truncated, info


class VectorEnv:
    def __init__(
            self,
            env_num: int,
            observation_space: gymnasium.Space,
            action_space: gymnasium.Space
    ):
        self.env_num = env_num
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def seed(self, seed):
        pass

    def close(self):
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()
