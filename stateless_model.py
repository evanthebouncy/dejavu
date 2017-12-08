# A General Purpose Stateless Agent For RL Under Policy Gradient #
# Work under Discrete Actions #

import numpy as np
import tensorflow as tf
import random

# useful for policy gradient under REINFORCE
# a trace is a list of (state, action, reward)
def get_discount_future_reward(trace):
  rewards = [tr[2] for tr in trace]
  disc_rewards = [0 for _ in range(len(rewards))]
  running_add = 0
  for t in reversed(range(0, len(rewards))):
    running_add = running_add * 0.98 + rewards[t]
    disc_rewards[t] = running_add
  return disc_rewards

def normalize_trace_batch(trace_batch):
  rewards = []
  for trace in trace_batch:
    # print trace
    for s, a, r in trace:
      rewards.append(r)
  meanz, stdz = np.mean(rewards), np.std(rewards)
  ret = []
  for trace in trace_batch:
    to_add = []
    for s, a, r in trace:
      # print r, meanz, stdz
      to_add.append((s,a, (r - meanz) / stdz))
    ret.append(to_add)

  print [asdf[2] for asdf in ret[0]]
  assert 0
  return ret

class StatesProccessor:
  def __init__(self, return_dim, proc_function):
    self.return_dim = return_dim
    self.proc = proc_function

class ActionDecoder:
  def __init__(self, actions):
    self.actions = actions
    self.action_dim = len(actions)

  def action_to_onehot_array(self, a):
    action_idx = self.actions.index(a)
    ret = np.array([0.0 for _ in range(self.action_dim)])
    ret[action_idx] = 1.0
    return ret

  def index_to_action(self, idx):
    return self.actions[idx]    

class StatelessAgent:

  # all arguments are self explainatory...
  # interface to the environment
  #   states_processor takes in a prefix of the partially generated trace and
  #   return a finite state representation
  #   action decoder takes in an action index and return an action
  def __init__(self, name, 
               states_processer, action_decoder,
               learning_rate = 0.001, num_hidden = 100, epi = 0.1):
    self.epi = epi
    self.name = name
    self.states_processer = states_processer
    self.action_decoder = action_decoder
    self.state_dim = states_processer.return_dim
    self.action_dim = action_decoder.action_dim

    self.graph = tf.Graph()
    self.session = tf.Session(graph = self.graph)
  
    with self.session.graph.as_default():
      # this is the input state
      self.input_state = tf.placeholder(tf.float32, [None, self.state_dim])
      # this is the roll-out-reward indexed by action on that particular state
      # used for training only
      self.roll_out_reward = tf.placeholder(tf.float32, [None, self.action_dim])

      # one layer of fc to predict the action
      self.hidden_state = tf.layers.dense(self.input_state, num_hidden, activation= tf.nn.relu)

      # predict the move
      self.prediction = tf.layers.dense(self.hidden_state, self.action_dim)
      self.pred_prob = tf.nn.softmax(self.prediction)

      # add a small number so it doesn't blow up (logp or in action selection)
      self.pred_prob = self.pred_prob + 1e-8

      # set up the cost function for training
      self.log_pred_prob = tf.log(self.pred_prob)
      self.objective = tf.reduce_mean(self.log_pred_prob * self.roll_out_reward)

      self.loss = -self.objective

      self.optimizer = tf.train.AdamOptimizer(learning_rate)
      self.train = self.optimizer.minimize(self.loss)

      initializer = tf.global_variables_initializer()
      self.session.run(initializer)

      self.saver = tf.train.Saver()

  def save_model(self, path):
    self.saver.save(self.session, path)
    print "model saved at ", path

  def restore_model(self, path):
    self.saver.restore(self.session, path)
    print "model restored  from ", path

  # only supports 1 state at a time, no batching plz
  # stocastic action, and some episilon for exploration
  # act takes in all the trace prefix, but the processor likely only use the last state
  def act(self, trace_prefix, cur_state, show_prob=False):
    inp = np.array([self.states_processer.proc(trace_prefix, cur_state)])
    the_action = self.session.run([self.pred_prob], {self.input_state: inp})[0][0]
    if show_prob:
      print "action prob ", the_action
    move_idx = np.random.choice(range(self.action_dim), p=the_action)
    if random.random() < self.epi:
      move_idx = np.random.choice(range(self.action_dim))
    return self.action_decoder.index_to_action(move_idx)
    

  # Use this for behavioural cloning
  def learn_supervised(self, supervised_trace_batch):
    batch_states = []
    batch_action_indexed_rewards = []

    for trace in supervised_trace_batch:
      trace_prefix = [trace[:i-1] if i > 0 else [] for i in range(len(trace))]
      states =  [self.states_processer.proc(trace_prefix[i], trace[i][0]) for i in range(len(trace))]
      actions = [tr[1] for tr in trace]
      for s, a in zip(states, actions):
        batch_states.append(s)
        batch_action_indexed_rewards.append(self.action_decoder.action_to_onehot_array(a))

    if batch_states == []: return

    batch_states = np.array(batch_states)
    batch_action_indexed_rewards = np.array(batch_action_indexed_rewards)

    loss_train = self.session.run([self.loss, self.train], {self.input_state: batch_states,
                                   self.roll_out_reward: batch_action_indexed_rewards})
    print "supervised loss ", loss_train[0]
    
  def learn_policy_grad(self, trace_batch):
    # trace_batch = normalize_trace_batch(trace_batch)
    batch_states = []
    batch_action_indexed_rewards = []

    for trace in trace_batch:
      states =  [self.states_processer.proc(trace[:i]) for i in range(len(trace))]
      actions = [tr[1] for tr in trace]
      disc_rewards = get_discount_future_reward(trace) 
      for s, a, r in zip(states, actions, disc_rewards):
        batch_states.append(s)
        # print r, a, self.xform_action(a)
        batch_action_indexed_rewards.append(r * self.action_decoder.action_to_onehot_array(a))

    if batch_states == []: return

    batch_states = np.array(batch_states)
    batch_action_indexed_rewards = np.array(batch_action_indexed_rewards)

    self.session.run([self.train], {self.input_state: batch_states,
                                    self.roll_out_reward: batch_action_indexed_rewards})

