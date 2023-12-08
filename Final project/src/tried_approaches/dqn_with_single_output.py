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
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out


def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx, nu):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx+nu))
    state_out1 = layers.Dense(16, activation="selu")(inputs) 
    state_out2 = layers.Dense(32, activation="selu")(state_out1) 
    state_out3 = layers.Dense(32, activation="selu")(state_out2) 
    state_out4 = layers.Dense(16, activation="selu")(state_out3)
    outputs = layers.Dense(1)(state_out4) 

    model = tf.keras.Model(inputs, outputs)
    return model


def update(x_batch, u_batch, reward_batch, x_next_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape:
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched.
        # Tensors can be manually watched by invoking the watch method on this context manager.
        x_next_batch = tf.transpose(x_next_batch, perm=[2, 0, 1, 3])
        xu_next_batch_q_values = []
        for i in range(len(x_next_batch)):
            xu_next = tf.concat([tf.concat([x_next_batch[i], tf.transpose(np2tf(np.array([np.eye(env.nu)[u]])), perm=[1, 0, 2])], axis=1) for u in range(env.nu)], axis=0)
            xu_next_batch_q_values.append(Q_target(xu_next, training=False))
        xu_next_batch_q_values = tf.stack(xu_next_batch_q_values)
        target_values = tf.math.reduce_min(xu_next_batch_q_values, axis=1)
        # Compute 1-step targets for the critic loss
        y = reward_batch + DISCOUNT * target_values
        # Compute batch of Values associated to the sampled batch of states
        x_batch = tf.transpose(tf.squeeze(x_batch, axis=0), perm=[1, 0, 2])
        u_batch = tf.concat([tf.transpose(np2tf(np.array([np.eye(env.nu)[int(u_batch[u])]])), perm=[1, 0, 2]) for u in range(len(u_batch))], axis=0)
        xu_batch = tf.concat([x_batch, u_batch], axis=1)
        Q_values = Q(xu_batch, training=True)
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_values))
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)
    # Update the critic backpropagating the gradients
    critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))


Experience = namedtuple('Experience', field_names=['state', 'action', 'cost', 'new_state'])


class ExperienceBuffer:

    def __init__(self, capacity):
        # Represents the buffer as a deque
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        # Adds the current experience to the buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Samples an index for each element in the batch
        indices = choice(len(self.buffer), batch_size, replace=False)

        # Extracts experience entries for each element in the batch
        # Each value returned by zip is a list of length batch_size
        states, actions, costs, next_states = zip(*[self.buffer[idx] for idx in indices])
        
        # Returns results as numpy arrays
        return np.array(states), np.array(actions), np.array(costs, dtype=np.float32), np.array(next_states)


class Agent:

    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.reset()

    def reset(self):
        # Restart the environment
        self.state = self.env.reset()

    def step(self, net, epsilon=0.0):
        # Sample the action randomly with probability epsilon
        if uniform() < epsilon:
            u = randint(self.env.nu)
        # Otherwise select action based on qvalues
        else:
            # Create a batch made of a single state
            # state_tensor = np2tf(np.array(np.eye(self.env.nx)[self.state]))  # one hot encoding ?
            # state_tensor = tf.expand_dims(state_tensor, 0)
            state_tensor = np2tf(np.array(self.state))

            # Get qvalues and select the index of the minimum
            xu = tf.concat([tf.concat([state_tensor, tf.transpose(np2tf(np.array([np.eye(self.env.nu)[u]])), perm=[1, 0, 2])], axis=1) for u in range(self.env.nu)], axis=0)
            q_values = Q_target(xu)
            u = np.argmin(tf2np(q_values))

        # Perform a step in the environment
        x_next, cost = self.env.step(u)

        # Save the new experience
        self.exp_buffer.append(Experience(self.state, u, cost, x_next))

        # Register the current state
        self.state = x_next
        
        return cost


def train(net, target_net, agent):
    frame_idx = 0
    # Epsilon starts from the initial value and is then annealed
    exploration_prob = EXPLORATION_PROB_START

    # Keep track of the cost-to-go history (for plot)
    h_ctg = []
    for k in range(NEPISODES):
        J = 0
        gamma_i = 1
        agent.reset()
        # simulate the system for maxEpisodeLength steps
        for i in range(MAX_EPISODE_LENGTH):
            cost = agent.step(net, exploration_prob)
            J += gamma_i * cost
            gamma_i *= DISCOUNT

            frame_idx +=1
            # Computes the current epsilon with linear annealing
            exploration_prob = max(MIN_EXPLORATION_PROB, exploration_prob - EXPLORATION_DECREASING_DECAY)

            # Continue to collect experience until the warmup finishes
            if len(agent.exp_buffer) < REPLAY_START_SIZE:
                continue

            # At regular intervals load the weights of the DQN into the target DQN
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                Q_target.set_weights(Q.get_weights())

            # Update DQN
            if frame_idx % UPDATE_DQN_FRAMES == 0:
                states, actions, rewards, next_states = buffer.sample(BATCH_SIZE)
                update(np2tf(states), np2tf(actions), np2tf(rewards), np2tf(next_states))

        h_ctg.append(J)
        if frame_idx % NPRINT == 0:
            print(f"Iter {k}, exploration prob {exploration_prob}")

    return h_ctg


def render_greedy_policy(net, env, x0=None):
    '''Roll-out from random state using greedy policy.'''
    x0 = x = env.reset(x0)
    cost_to_go = 0.0
    gamma_i = 1
    for i in range(MAX_EPISODE_LENGTH):
        # Create a batch made of a single state
        # state_tensor = np2tf(np.array(np.eye(self.env.nx)[x]))  # one hot encoding ?
        # state_tensor = tf.expand_dims(state_tensor, 0)
        state_tensor = np2tf(np.array(x))

        # Get qvalues and select the index of the minimum
        xu = tf.concat([tf.concat([state_tensor, tf.transpose(np2tf(np.array([np.eye(env.nu)[u]])), perm=[1, 0, 2])], axis=1) for u in range(env.nu)], axis=0)
        q_values = Q_target(xu)
        u = np.argmin(tf2np(q_values))

        x, cost = env.step(u)
        print(cost)
        cost_to_go += gamma_i * cost
        gamma_i *= DISCOUNT
        env.render()

    print(f"Real cost to go of state {x0} : {cost_to_go}")


if __name__=='__main__':
    ### Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


    ### Hyper paramaters
    NEPISODES = 10000                       # Number of training episodes
    NPRINT = 1000                           # print something every NPRINT episodes
    MAX_EPISODE_LENGTH = 200                # Max episode length
    LEARNING_RATE = 0.001                   # learning rate
    DISCOUNT = 0.99                         # Discount factor 
    PLOT = True                             # Plot stuff if True
    EXPLORATION_PROB_START = 1.0            # initial exploration probability of eps-greedy policy
    EXPLORATION_DECREASING_DECAY = 5e-6     # exploration decay for exponential decreasing
    MIN_EXPLORATION_PROB = 0.001            # minimum of exploration probability
    BATCH_SIZE = 32                         # size of the batch
    REPLAY_SIZE = 50000                     # Size of the replay buffer
    REPLAY_START_SIZE = 50000               # Warmup frames for the replay buffer
    UPDATE_DQN_FRAMES = 10                  # Number of steps at which updated the weights of the DQN
    SYNC_TARGET_FRAMES = 1000               # Number of steps at which transfer weights from the DQN to the target DQN


    ### Environment
    # nq = 51   # number of discretization steps for the joint angle q
    # nv = 21   # number of discretization steps for the joint velocity v
    nu = 11   # number of discretization steps for the joint torque u
    # env = DPendulum(nq, nv, nu)
    env = Pendulum(nu=nu)

    # Create critic and target NNs
    Q = get_critic(env.nx, env.nu)
    Q_target = get_critic(env.nx, env.nu)

    # Load NN weights from file
    Q.load_weights("qweightsv4.h5")

    # Set initial weights of targets equal to those of actor and critic
    Q_target.set_weights(Q.get_weights())

    # Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    # Trains the network
    h_ctg = train(Q, Q_target, agent)  # return the average cost to go and plot it

    print("Average cost: %.3f" % (sum(h_ctg)/NEPISODES))
    plt.plot( np.cumsum(h_ctg)/range(1, NEPISODES + 1) )
    plt.title("Average cost-to-go")
    plt.show()

    # Save NN weights to file (in HDF5)
    Q.save_weights("qweightsv4.h5")  # TODO: we should save the weigths of the best performing net

    render_greedy_policy(Q, env)
