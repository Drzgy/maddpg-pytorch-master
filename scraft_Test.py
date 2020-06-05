import numpy as np
from smac.env import StarCraft2Env
from scraft_ENV_test import *
import argparse
import torch
import time
import os
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = True


def run(config):
    # model_dir = Path('./models') / config.env_id / config.model_name
    # if not model_dir.exists():
    run_num = 10
    # else:
    #    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
    #                     model_dir.iterdir() if
    #                     str(folder.name).startswith('run')]
    #    if len(exst_run_nums) == 0:
    #        run_num = 1
    #    else:
    #        run_num = max(exst_run_nums) + 1
    # curr_run = 'run%i' % run_num
    # run_dir = model_dir / curr_run
    # log_dir = run_dir / 'logs'
    log_dir = 'checkpoints0605_4/'
    run_dir = log_dir + 'logs/'
    # os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    # change env
    env = SCraftAdapter(map_name='3m', seed=123,step_mul=8, difficulty='7', game_version='latest', replay_dir="replay/")
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp for obsp in env.observation_space],
                                 [acsp for acsp in env.action_space])
    t = 0

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        stop = False
        obs = env.reset()
        # if stop:
        #     stop = False
        #     obs = env.reset()
        # else:
        #     obs=env._get_obs()

        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(
            config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        episode_length = 0
        while not stop:
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)


            actions=[]
            agent_actions=np.zeros([len(env.action_space),env.action_space[0]])
            # add avail_agent
            avail_actions=np.array(env.get_avail_actions())
            for agent_i in range(len(torch_agent_actions)):
                agent_action = env.get_avail_agent_actions(agent_i)
                agent_action=[0 if agent_action[i]==0 else torch_agent_actions[agent_i].data.numpy()[0][i] for i in range(len(agent_action))]
            # add argmax
                actions.append(np.argmax(agent_action))
            # new actions
                agent_actions[agent_i][actions[agent_i]]=1


            # torch_agent_actions=[(if agent_avail_actions  for action in ac.data.numpy()) for ac in torch_agent_actions]
            # convert actions to numpy arrays

            # rearrange actions to be per environment

            # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            stop = dones[0][0]
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones,avail_actions)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
            episode_length += 1
        ep_rews = replay_buffer.get_average_rewards(
            episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

    #     if ep_i % config.save_interval < config.n_rollout_threads:
    #         os.makedirs(run_dir / 'incremental', exist_ok=True)
    #         maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
    #         maddpg.save(run_dir / 'model.pt')
    #
    # maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir, '/summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="simple_spread")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="MADDPG")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    #parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
