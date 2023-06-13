import os
import random
from typing import Tuple, Dict, Any, Optional
import numpy as np
from imageio import mimsave
import gym
import vizdoomgym
import vizdoom as vzd
import cv2
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.monitor import Monitor


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self,
                 env: gym.Env = None,
                 rescale_factor: float = 1.0):
        super().__init__(env)

        self.rescale_factor = rescale_factor
        self.original_shape = self.observation_space.shape
        self.new_shape = (int(self.original_shape[0] * rescale_factor), int(self.original_shape[1] * rescale_factor), self.original_shape[2])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.new_shape, dtype=np.uint8)

    def observation(self, obs) -> np.ndarray:
        resized_screen = cv2.resize(obs, (self.new_shape[1], self.new_shape[0]), interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.new_shape)
        return new_obs


class RepeatAction(gym.Wrapper):
    def __init__(self,
                 env: gym.Env =None,
                 repeat: int = 4,
                 render_screen: bool =False,
                 record_video: bool =False):
        super().__init__(env)

        self.repeat = repeat
        self.render_screen = render_screen
        self.record_video = record_video
        self.images = []
        self.kills = 0

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        t_reward = 0.0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward

            if self.env.game.get_game_variable(vzd.GameVariable.KILLCOUNT) > self.kills:
                t_reward += 50
                self.kills = self.env.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            
            if self.render_screen:
                self.env.render()
            
            if not done and self.record_video:
                img = self.env.game.get_state().screen_buffer
                self.images.append(img)
            
            if done:
                break
        
        return obs, t_reward, done, info

    def reset(self) -> None:
        return self.env.reset()


def create_env(scenario: str = 'VizdoomBasic-v0',
               repeat: int = 4,
               rescale_factor: float = 0.5,
               render_screen: bool = False,
               record_video: bool = False) -> gym.Env:
    
    env = gym.make(scenario)
    env = RepeatAction(env, repeat, render_screen, record_video)
    env = PreprocessFrame(env, rescale_factor)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecTransposeImage(env)

    return env


def save_gif(env: gym.Env,
             file_name: str,
             fps: int = 30):
    
    images = np.array(env.get_attr('images')[0])
    num_frames = np.array(env.get_attr('images')[0]).shape[0]
    list_images = [np.moveaxis(images[i,...],0,-1) for i in range(num_frames)]

    mimsave(f'./videos/{file_name}.gif', list_images, fps=fps)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def seed_everything(seed: int,
                    env: Optional[gym.Env] = None,
                    use_deterministic_algos: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms(use_deterministic_algos)
    random.seed(seed)

