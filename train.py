from stable_baselines3 import PPO
from utils import create_env, seed_everything, make_dir
from callbacks import TensorBoardCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


def train_ppo(model_name: str = 'ppo',
              env_name: str = 'VizdoomCorridor-v0',
              policy: str = 'CnnPolicy',
              frame_repeat: int = 4,
              rescale_factor: float = 0.5,
              train_timesteps: int = 10000,
              eval_frequency: int = 5000,
              eval_episodes: int = 5):
    
    seed_everything(42)
    make_dir('saved_models')

    env = create_env(env_name, repeat=frame_repeat, rescale_factor=rescale_factor)

    model = PPO(policy, env, verbose=1, tensorboard_log=f"./tensorboard/{env_name}/{model_name}/")

    eval_callback = EvalCallback(env,
                                 best_model_save_path=f"./saved_models/{env_name}/checkpoint_models",
                                 eval_freq=eval_frequency,
                                 n_eval_episodes=eval_episodes,
                                 deterministic=True,
                                 render=False)
    
    callback = TensorBoardCallback(average_reward=True)

    print("Training...")
    model.learn(total_timesteps=train_timesteps, callback=[eval_callback, callback])
    model.save(f"./saved_models/{env_name}/{model_name}")

    print("Evaluation...")

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward {mean_reward}\tstd_reward {std_reward}")


if __name__ == "__main__":
    train_ppo()
