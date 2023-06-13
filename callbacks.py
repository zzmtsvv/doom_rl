from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCallback(BaseCallback):
    def __init__(self,
                 average_reward: bool = True,
                 verbose: int = 1):
        super().__init__(verbose)

        self.average_reward = average_reward
        self.total_reward = 0
        self.episode_length = 0
    
    def _on_step(self) -> bool:
        if self.average_reward:
            unscaled_reward = self.training_env.get_original_reward()[0]
            self.total_reward += unscaled_reward
            self.episode_length += 1
        
        return True
    
    def _on_rollout_end(self) -> None:
        if self.average_reward:
            self.avg_policy_return = self.total_reward / self.episode_length
            self.logger.record("train/average_reward", self.avg_policy_return)
        
        self.total_reward = 0
        self.episode_length = 0
