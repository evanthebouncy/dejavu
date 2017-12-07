# Sample a Dagger Trace Aggregator Thing

from stateless_model import *

class Dagger:
 
  def __init__(self, env, env_actions):
    self.trace_gen = TraceGenerator(env, env_actions)

  def generate_dagger_trace(self, student, student_states_proc, 
                                  expert,  expert_states_proc):
    student_trace = self.trace_gen.generate_trace(student, student_states_proc, start_state)
    expert_augmented_trace = []
    for i, tr in enumerate(student_trace):

     
