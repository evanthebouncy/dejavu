# Sample a Dagger Trace Aggregator Thing
from stateless_model import *

def generate_trace(env, agent, start_state=None, n=200, do_render=False):
  env.reset() if start_state is None else env.restore_full_state(start_state)
  cur_state = env._get_obs()
  return_trace = []

  for i in range(n):
    action = agent.act(return_trace, cur_state)
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
    s,a,r = tr
    trace_prefix = student_trace[:i-1] if i > 0 else []
    expert_a = expert.act(trace_prefix, s)
    expert_augmented_trace.append( (s,expert_a,r) )
  return expert_augmented_trace

class Aggregater:

  def __init__(self, max_buffer_size):
    self.max_buffer_size = max_buffer_size
    self.buf = []
         
  def add_new_trace(self, env, student, expert, start_state):
    student_trace = generate_trace(env, student, start_state)
    total_reward = sum([sss[2] for sss in student_trace])
    new_dagger_trace = generate_dagger_trace(env, student, expert, start_state)

    # this is only because expert might not be optimal, so if student trace has good reward
    # we should learn from it
    print total_reward
    if total_reward >= 0.0:
      print "student trace was good it added"
      self.buf.append(student_trace)
    else:
      print "student trace was no good added dagger"
      self.buf.append(new_dagger_trace)
 
    if len(self.buf) > self.max_buffer_size:
      rand_id = random.choice(range(len(self.buf)))
      del self.buf[rand_id]

  def sample_trace(self):
    return random.choice(self.buf)

     
