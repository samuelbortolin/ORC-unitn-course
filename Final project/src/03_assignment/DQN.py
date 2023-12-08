from collections import namedtuple, deque
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint, uniform, choice
from pendulum import Pendulum
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
import time


np_config.enable_numpy_behavior()

 
def np2tf(y):
    ''' Convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out


def tf2np(y):
    ''' Convert from tensorflow to numpy '''
    out = tf.squeeze(y).numpy()
    return out


def get_critic(nx, nu, nj):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx))
    state_out1 = layers.Dense(16, activation="selu")(inputs)
    state_out2 = layers.Dense(32, activation="selu")(state_out1)
    state_out3 = layers.Dense(32, activation="selu")(state_out2)
    last_layer_dimension = 16
    if nj > 1:
        if nu > 7:
            last_layer_dimension = 64
        else:
            last_layer_dimension = 32
    state_out4 = layers.Dense(last_layer_dimension, activation="selu")(state_out3)  # For 1 joint use 16 units, with 2 joints use 32 units for nu=7 and 64 units for nu=9
    outputs = layers.Dense(nu**nj)(state_out4)

    model = tf.keras.Model(inputs, outputs)
    return model


def update(x_batch, u_batch, cost_batch, x_next_batch, discount):
    ''' Update the weights of the Q network using the specified batch of data '''
    # All input batch elements are tf tensors
    with tf.GradientTape() as tape:
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched.
        # Tensors can be manually watched by invoking the watch method on this context manager.
        x_next_batch = tf.transpose(tf.squeeze(x_next_batch, axis=0), perm=[1, 0, 2])
        target_values = tf.math.reduce_min(Q_target(x_next_batch, training=False), axis=1, keepdims=True)
        # Compute 1-step targets for the critic loss
        y = cost_batch + discount * target_values
        # Create indexed control batch in order to use tf.gather_nd to pick the right output from the network
        u_batch_indexed = []
        for i in range(len(u_batch)):
            u_batch_indexed.append([i, int(u_batch[i])])
        u_batch_indexed = tf.stack(u_batch_indexed)
        # Compute batch of Values associated to the sampled batch of states
        x_batch = tf.transpose(tf.squeeze(x_batch, axis=0), perm=[1, 0, 2])
        Q_values = tf.expand_dims(tf.gather_nd(Q(x_batch, training=True), u_batch_indexed), 1)
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_values))
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)
    # Update the critic backpropagating the gradients
    critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))


Experience = namedtuple('Experience', field_names=['state', 'control', 'cost', 'next_state'])


class ExperienceBuffer:

    def __init__(self, capacity):
        # Represent the buffer as a deque
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        # Return the length of the buffer
        return len(self.buffer)

    def append(self, experience):
        # Add an experience element to the buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample an index for each element in the batch
        indices = choice(len(self.buffer), batch_size, replace=False)

        # Extract experience entries for each element in the batch
        # Each value returned by zip is a list of length batch_size
        states, controls, costs, next_states = zip(*[self.buffer[idx] for idx in indices])
        
        # Return results as numpy arrays
        return np.array(states), np.array(controls), np.array(costs, dtype=np.float32), np.array(next_states)


class Agent:

    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.reset()

    def reset(self):
        # Restart the environment
        self.state = self.env.reset()

    def step(self, exploration_prob):
        # Sample the control randomly with probability exploration_prob
        if uniform() < exploration_prob:
            u = np.array([randint(self.env.nu) for _ in range(self.env.nj)])
            u_index = 0
            for i in range(self.env.nj):
                u_index += self.env.nu ** i * u[i]
        # Otherwise select control based on qvalues
        else:
            # Create a batch made of a single state
            state_tensor = np2tf(np.array(self.state))

            # Get qvalues and select the index of the minimum for each joint
            q_values = Q(state_tensor)
            u_index = np.argmin(tf2np(q_values))
            u_index_copy = u_index.copy()
            u = []
            for i in reversed(range(self.env.nj)):
                u.append(u_index_copy // (self.env.nu ** i))
                u_index_copy -= self.env.nu ** i * u[self.env.nj - 1 - i]
            u.reverse()
            u = np.array(u)

        # Perform a step in the environment
        self.env.reset(self.state)
        x_next, cost = self.env.step(u)

        # Save the new experience
        self.exp_buffer.append(Experience(self.state, u_index, cost, x_next))

        # Register the current state
        self.state = x_next
        return cost


def train(agent,
          exploration_prob_start,
          exploration_decreasing_decay,
          min_exploration_prob,
          n_episodes,
          max_episode_length,
          discount,
          update_dqn_steps,
          batch_size,
          replay_start_size,
          sync_target_steps,
          n_print,
          use_evaluate_greedy_policy):
    ''' Train the network '''
    steps_idx = 0
    # Epsilon starts from the initial value and is then annealed
    exploration_prob = exploration_prob_start

    # Keep track of the cost-to-go history (for plot)
    h_ctg = []
    
    # Keep track of the best weights and of the best cost so far
    best_weights_so_far = Q.get_weights()
    best_cost_to_go_so_far = -np.inf
    try:
        for k in range(n_episodes):
            J = 0
            gamma_i = 1
            agent.reset()

            if k % n_print == 0:
                print(f"Iter {k}, exploration prob {exploration_prob}")

            # Simulate the system for max_episode_length steps
            for _ in range(max_episode_length):
                cost = agent.step(exploration_prob)
                J += gamma_i * cost
                gamma_i *= discount

                steps_idx +=1
                # Compute the current epsilon with linear decreasing
                exploration_prob = max(min_exploration_prob, exploration_prob - exploration_decreasing_decay)

                # Continue to collect experience until the warmup finishes
                if len(agent.exp_buffer) < replay_start_size:
                    continue

                # Load the weights of the DQN into the target DQN every sync_target_steps steps
                if steps_idx % sync_target_steps == 0:
                    Q_target.set_weights(Q.get_weights())

                # Update DQN every update_dqn_steps steps
                if steps_idx % update_dqn_steps == 0:
                    states, controls, costs, next_states = agent.exp_buffer.sample(batch_size)
                    update(np2tf(states), np2tf(controls), np2tf(costs), np2tf(next_states), discount)

                    if use_evaluate_greedy_policy:
                        evaluated_cost_to_go = evaluate_greedy_policy(agent.env, max_episode_length, discount)
                        if evaluated_cost_to_go < best_cost_to_go_so_far:
                            best_weights_so_far = Q.get_weights()
                            best_cost_to_go_so_far = evaluated_cost_to_go

            h_ctg.append(J)

    except KeyboardInterrupt:
        print(f"Stopped at iter {k}, exploration prob {exploration_prob}")

    if use_evaluate_greedy_policy:
        return h_ctg, best_weights_so_far
    else:
        return h_ctg, Q.get_weights()


def evaluate_greedy_policy(env, max_episode_length, discount, angle_discretization=8):
    ''' Evalute Q network using greedy policy on some predefined states '''
    x = np.array([env.reset(np.vstack([np.array([[i*2*np.pi/angle_discretization-np.pi] for _ in range(env.nq)]), np.array([[0] for _ in range(env.nv)])])) for i in range(angle_discretization)])
    # x = np.append(x, np.array([env.reset(np.vstack([np.array([[i*2*np.pi/angle_discretization-np.pi] for _ in range(env.nq)]), np.array([[0.5] for _ in range(env.nv)])])) for i in range(angle_discretization)]), axis=0)
    # x = np.append(x, np.array([env.reset(np.vstack([np.array([[i*2*np.pi/angle_discretization-np.pi] for _ in range(env.nq)]), np.array([[-0.5] for _ in range(env.nv)])])) for i in range(angle_discretization)]), axis=0)

    evaluated_cost_to_go = 0.0
    gamma_i = 1
    for _ in range(max_episode_length):
        # Create a batch of all states
        state_tensor = tf.transpose(tf.squeeze(np2tf(x), axis=0), [1, 0, 2])

        # Get qvalues and select the index of the minimum for each joint
        q_values = Q(state_tensor)
        u_index = np.argmin(tf2np(q_values), axis=1)
        u = []
        for i in reversed(range(env.nj)):
            u.append(u_index // (env.nu ** i))
            u_index -= env.nu ** i * u[env.nj - 1 - i]
        u.reverse()
        u = np.transpose(np.array(u))

        for i in range(len(x)):
            env.reset(x[i])
            x[i], cost = env.step(u[i])
            evaluated_cost_to_go += gamma_i * cost
        gamma_i *= discount

    return evaluated_cost_to_go


def render_greedy_policy(env, max_episode_length, discount, x0=None):
    ''' Roll-out from random state using greedy policy given by Q network '''
    x0 = x = env.reset(x0)
    cost_to_go = 0.0
    gamma_i = 1
    for _ in range(max_episode_length):
        # Create a batch made of a single state
        state_tensor = np2tf(np.array(x))

        # Get qvalues and select the index of the minimum for each joint
        q_values = Q(state_tensor)
        u_index = np.argmin(tf2np(q_values))
        u = []
        for i in reversed(range(env.nj)):
            u.append(u_index // (env.nu ** i))
            u_index -= env.nu ** i * u[env.nj - 1 - i]
        u.reverse()
        u = np.array(u)

        x, cost = env.step(u)
        cost_to_go += gamma_i * cost
        gamma_i *= discount
        env.render()

    print(f"Real cost to go of state {x0} : {cost_to_go}")


if __name__=='__main__':

    ### Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### Environment
    nj = 2                                                  # Number of joints of the pendolum
    nu = 5                                                  # Number of discretization steps for the joint torque u (For 1 joint use 11 discretization steps, with 2 joints use 5 discretization steps but possible to use also 7 or 9 discretization steps)
    umax = 4.0 if nj > 1 else 2.0                           # Maximum torque for the joints of the pendolum (For 1 joint use 2.0, with 2 joints use 4.0)

    ### Hyper paramaters
    NEPISODES = 20000 if nj > 1 else 10000                  # Number of training episodes (For 1 joint use 10000 episodes, with 2 joints use 20000 episodes)
    MAX_EPISODE_LENGTH = 500 if nj > 1 else 200             # Max episode length (For 1 joint use 200 episode length, with 2 joints use 500 episode length)
    NPRINT = 50                                             # Print something every NPRINT episodes
    DISCOUNT = 0.99                                         # Discount factor
    EXPLORATION_PROB_START = 1.0                            # Initial exploration probability of eps-greedy policy
    EXPLORATION_DECREASING_DECAY = 5e-7 if nj > 1 else 5e-6 # Exploration decay for exponential decreasing (For 1 joint use 5e-6 decay, with 2 joints use 5e-7 decay)
    MIN_EXPLORATION_PROB = 0.001                            # Minimum of exploration probability
    REPLAY_SIZE = 50000                                     # Size of the replay buffer
    REPLAY_START_SIZE = 50000                               # Warmup steps for the replay buffer
    UPDATE_DQN_STEPS = 10                                   # Number of steps at which updated the weights of the DQN
    BATCH_SIZE = 32                                         # Size of the batch used for training
    LEARNING_RATE = 0.001                                   # Learning rate for the Adam optimizer
    SYNC_TARGET_STEPS = 1000                                # Number of steps at which transfer weights from the DQN to the target DQN
    TRAINING = False                                        # Train the network if True
    USE_EVALUATE_GREEDY_POLICY = False                      # Use evaluate_greedy_policy method to get the an evaluation of the policy for the current weights if True

    # Create environmnet
    env = Pendulum(nj=nj, nu=nu, umax=umax)

    # Create buffer and agent
    exp_buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, exp_buffer)

    # Create critic and target NNs (Note that tensorflow stuff is treated as global)
    Q = get_critic(env.nx, env.nu, env.nj)
    Q_target = get_critic(env.nx, env.nu, env.nj)

    weights_file = "2joints_5nu.h5"   # For 1 joint use 1joint_11nu.h5, with 2 joints with nu=5 use 2joints_5nu.h5, with nu=7 use 2joints_7nu.h5 and with nu=9 use 2joints_9nu.h5

    # Load NN weights from file
    Q.load_weights(weights_file)

    # Set initial weights of targets equal to those of actor and critic
    Q_target.set_weights(Q.get_weights())

    # Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    if TRAINING:
        # Train the network. Returns the average cost to go to plot it and the best weights
        h_ctg, best_weights = train(agent,
                                    EXPLORATION_PROB_START,
                                    EXPLORATION_DECREASING_DECAY,
                                    MIN_EXPLORATION_PROB,
                                    NEPISODES,
                                    MAX_EPISODE_LENGTH,
                                    DISCOUNT,
                                    UPDATE_DQN_STEPS,
                                    BATCH_SIZE,
                                    REPLAY_START_SIZE,
                                    SYNC_TARGET_STEPS,
                                    NPRINT,
                                    USE_EVALUATE_GREEDY_POLICY)

        print("Average cost: %.3f" % (sum(h_ctg)/len(h_ctg)))
        plt.plot( np.cumsum(h_ctg)/range(1, len(h_ctg) + 1) )
        plt.title("Average cost-to-go")
        plt.show()

        # Save NN weights to file
        Q.set_weights(best_weights)
        Q.save_weights(weights_file)

    render_greedy_policy(env, MAX_EPISODE_LENGTH, DISCOUNT)
