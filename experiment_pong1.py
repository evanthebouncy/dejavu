from stateless_model import *
import gym.envs.atari
import gym
import random
from subsample import *
from utils import preprocess, get_random_pong_state

def state_processor1(states):
  if len(states) < 2:
    # return np.zeros([800+100*2])
    return np.zeros([100*2])

  ob1, ob2 = preprocess(states[-1]), preprocess(states[-2])

  pad1 = ob1[:, 65:70]
  pad2 = ob2[:, 65:70]
  ob1, ob2 = dim_reduce(ob1, 3), dim_reduce(ob2, 3)
  together = np.concatenate([ob1, ob2], axis=1)
  
  pad1_flat = np.reshape(pad1, [400])
  pad2_flat = np.reshape(pad2, [400])

  together_flat = np.reshape(together, [100*2])
  together_with_prev = np.concatenate([together_flat, pad1_flat, pad2_flat])
  # return together_with_prev
  return together_flat

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
start_state = env.clone_full_state()

state_dim = 200
action_dim = 2
collector = TraceGenerator(env, [2,3])
stateless_agent = StatelessAgent("bob", state_dim, action_dim)

ctr = 0
times_explore = 10
while True:
  ctr += 1
  do_render = True if ctr % 10 == 0 else False
  print ctr, do_render

  if ctr % 100 == 0:
    stateless_agent.save_model("models/pongpong.ckpt")

  # train the discriminator
#  planner_sa = [sample_planner_sa()]
#  stateless_agent.learn_supervised(planner_sa)

  # create a batch of trace in it
  agent_trace_batch = [collector.generate_trace(stateless_agent, state_processor1, 
                                                # get_random_pong_state(env, start_state),
                                                start_state,
                                                do_render = do_render)
                       for _ in range(times_explore)]

  for agr in agent_trace_batch:
    rewards = 0.0
    for s,a,r in agr:
      rewards += r
    print "explore trace reward ", rewards

  stateless_agent.learn_policy_grad(agent_trace_batch)

  

  
