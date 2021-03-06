import numpy as np
from smac.env import StarCraft2Env
from gym import spaces


class SCraftAdapter():

    def __init__(self, map_name, seed=123, step_mul=8, difficulty='7', game_version=None, replay_dir=""):
        self.env = StarCraft2Env(map_name=map_name,
                                 step_mul=step_mul,
                                 seed=123,
                                 difficulty=difficulty,
                                 game_version=game_version,
                                 replay_dir=replay_dir)
        env_info = self.env.get_env_info()
        self.observation_space = [env_info["obs_shape"] for i in
                                  range(env_info["n_agents"])]
        self.action_space = [env_info["n_actions"] for i in
                             range(env_info["n_agents"])]
        self.agent_types = ['agent' for i in range(env_info["n_agents"])]

    def reset(self):
        self.env.reset()
        return self._get_obs()

    # action_n: list[int]
    def step(self, action_n):
        info_n = {}
        actions = [int(action) for action in action_n]
        avail_actions = self.env.get_avail_actions()

        actions = [actions[i] if avail_actions[i][actions[i]] == 1 else np.argmax(avail_actions[i]) for i in
                   range(len(actions))]
        reward, done, info = self.env.step(actions)
        reward_n = np.array([[reward for i in range(len(actions))]])
        done_n = np.array([[done for i in range(len(actions))]])
        obs_n = np.array([self.env.get_obs()])
        info_n['n'] = info
        # obs_n: list[nd.array] reward_n: list[float] done_n:list[bool]
        return obs_n, reward_n, done_n, info_n

    def _get_obs(self):
        return np.array([self.env.get_obs()])

    def _get_done(self):
        pass

    def _get_reward(self):
        pass

    # output: list[list[int onehot coding] int agent id]
    def get_avail_actions(self):
        return self.env.get_avail_actions()

    # output: list[int onehot coding]  ex:[0,1,1,1,1,0,0,0,0]
    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def close(self):
        self.env.close()

