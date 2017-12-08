from stateless_model import *
import gym.envs.atari
import gym
import random
from subsample import *
from pong_utils import *
from dagger import *

class UpDownExpert:
  def act(self, trace_prefix, state):
#    states = [tr[0] for tr in trace_prefix]
    if len(trace_prefix) < 1:
      return random.choice([2,3])
    ob = state
    ball = get_ball(ob)
    pad = get_our_paddle(ob)
    if ball is None:
      return random.choice([2,3])
    ball_y, pad_y = ball[1], pad[1]
    is_up = pad_y > ball_y + 3
    is_down = pad_y < ball_y - 3
    if is_up: return 2
    if is_down: return 3  
    # if not we do counter move
    last_move = trace_prefix[-1][1]
    if last_move == 2: return 3
    if last_move == 3: return 2

expert_dumbkoft = UpDownExpert()


# take in a trace prefix and return the state (for the last 2 or somting liek that)
def state_processor1(trace_prefix, cur_state):
  states = [tr[0] for tr in trace_prefix]
  if len(states) < 1:
    return np.zeros([400])

  ob1, ob2 = preprocess(cur_state), preprocess(states[-1])
  if ob1 is None or ob2 is None:
    return np.zeros([400])
  else:

    pad1, pad2 = padpad(ob1), padpad(ob2)

    ob1, ob2 = dim_reduce(ob1, 3), dim_reduce(ob2, 3)
    together = np.concatenate([ob1, ob2], axis=1)

    pad_diff = np.reshape(pad1 - pad2, [200])

    together_flat = np.reshape(together, [100*2])
    together = np.concatenate([together_flat, pad_diff])
    return together

proc1 = StatesProccessor(400, state_processor1)
action_decoder = ActionDecoder([2,3])

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
start_state = env.clone_full_state()

stateless_agent = StatelessAgent("bob",proc1, action_decoder)

agg = Aggregater(400)

ctr = 0
times_explore = 10

generate_trace(env, expert_dumbkoft, 
               get_random_pong_state(env, start_state), n=1000, do_render=True)
while True:
  ctr += 1
  print ctr
  if ctr % 10 == 0:
    generate_trace(env, stateless_agent, 
                   get_random_pong_state(env, start_state), n=1000, do_render=True)

  if ctr % 100 == 0:
    stateless_agent.save_model("models/pongpong.ckpt")

  for _ in range(10):
    agg.add_new_trace(env, stateless_agent, expert_dumbkoft, 
                      get_random_pong_state(env,start_state))

  trace_batch = [agg.sample_trace() for _ in range(10)] 
  stateless_agent.learn_supervised(trace_batch)

    
  

  
