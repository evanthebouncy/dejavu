# Sample a Dagger Trace Aggregator Thing
from stateless_model import *

def generate_trace(env, agent, start_state=None, n=200, do_render=False):
  env.reset() if start_state is None else env.restore_full_state(start_state)
  cur_state = env._get_obs()
  return_trace = []
  for i in range(n):
    action = agent.act(return_trace)
    next_state, reward, done, comments = env.step(action)
    if do_render:
      env.render()
    return_trace.append((cur_state, action, reward))
    cur_state = next_state
    if done: break

  return return_trace

def generate_dagger_trace(env, student, expert, start_state):
  student_trace = generate_trace(env, student, start_state)
  expert_augmented_trace = []
  for i, tr in enumerate(student_trace):
    trace_prefix = student_trace[:i]
    expert_a = expert.act(trace_prefix)
    s,a,r = tr
    expert_augmented_trace.append( (s,expert_a,r) )
  return expert_augmented_trace

class Aggregater:

  def __init__(self, max_buffer_size):
    self.max_buffer_size = max_buffer_size
    self.buf = []
         
  def add_new_trace(self, env, student, expert, start_state):
    new_dagger_trace = generate_dagger_trace(env, student, expert, start_state)
    self.buf.append(new_dagger)
    if len(self.buf) > self.max_buffer_size:
      rand_item = random.choice(self.buf)
      self.buf.remove(rand_item)

  def sample_trace(self):
    return random.choice(self.buf)

     
