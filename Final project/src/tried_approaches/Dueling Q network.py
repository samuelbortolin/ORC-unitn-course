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


def get_critic(nx, nu, nj):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx))  # TODO: is it ok to change the original net?
    state_out1 = layers.Dense(16, activation="selu")(inputs)
    state_out2 = layers.Dense(32, activation="selu")(state_out1)
    state_out3 = layers.Dense(32, activation="selu")(state_out2)
    
    value_out1 = layers.Dense(16, activation="selu")(state_out3)
    value_out2 = layers.Dense(1)(value_out1)
    
    advantage_out1 = layers.Dense(64, activation="selu")(state_out3)
    advantage_out2 = layers.Dense(nu**nj)(advantage_out1)
    advantage_out3 = layers.Lambda(lambda a: a[:, :] - tf.reduce_mean(a[:, :], keepdims=True, axis=1),
                           output_shape=(nu**nj))(advantage_out2)
    outputs = layers.Add()([value_out2, advantage_out3])
    # outputs = layers.Dense(nu**nj)(state_out4)

    model = tf.keras.Model(inputs, outputs)
    return model


def update(x_batch, u_batch, reward_batch, x_next_batch):  # adjusted update method in order to support multiple joints
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape:
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched.
        # Tensors can be manually watched by invoking the watch method on this context manager.
        x_next_batch = tf.transpose(tf.squeeze(x_next_batch, axis=0), perm=[1, 0, 2])
        target_indexes = tf.math.argmin(Q(x_next_batch, training=False), axis=1)
        target_indexes_indexed = []
        for i in range(len(target_indexes)):
            target_indexes_indexed.append([i, int(target_indexes[i])])
        target_values = tf.gather_nd(Q_target(x_next_batch, training=False), target_indexes_indexed)
        
        # target_values = tf.math.reduce_min(Q_target(x_next_batch, training=False), axis=1, keepdims=True)
        # Compute 1-step targets for the critic loss
        y = reward_batch + DISCOUNT * target_values
        # Create indexed action batch in order to use tf.gather_nd to pick the right output from the network
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
            u = np.array([randint(self.env.nu) for _ in range(self.env.nj)])
            u_index = 0
            for i in range(self.env.nj):
                u_index += self.env.nu ** i * u[i]
        # Otherwise select action based on qvalues
        else:
            # Create a batch made of a single state
            state_tensor = np2tf(np.array(self.state))

            # Get qvalues and select the index of the minimum for each joint
            q_values = net(state_tensor)
            u_index = np.argmin(tf2np(q_values))
            u = []
            for i in reversed(range(self.env.nj)):
                u.append(u_index // (self.env.nu ** i))
                u_index -= self.env.nu ** i * u[self.env.nj - 1 - i]
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


def train(net, target_net, agent):
    frame_idx = 0
    # Epsilon starts from the initial value and is then annealed
    exploration_prob = EXPLORATION_PROB_START

    # Keep track of the cost-to-go history (for plot)
    h_ctg = []
    
    # Save the best weights
    best_weights_so_far = net.get_weights()
    best_cost_to_go_so_far = -np.inf
    try:
        for k in range(NEPISODES):
            J = 0
            gamma_i = 1
            agent.reset()
            # simulate the system for maxEpisodeLength steps
            for _ in range(MAX_EPISODE_LENGTH):
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
                    states, actions, rewards, next_states = agent.exp_buffer.sample(BATCH_SIZE)
                    update(np2tf(states), np2tf(actions), np2tf(rewards), np2tf(next_states))
                    
                    """evaluated_cost_to_go = evalutate_greedy_policy(net, env)
                    if evaluated_cost_to_go < best_cost_to_go_so_far:
                        best_weights_so_far = net.get_weights()
                        best_cost_to_go_so_far = evaluated_cost_to_go"""
    
            h_ctg.append(J)
            if frame_idx % NPRINT == 0:
                print(f"Iter {k}, exploration prob {exploration_prob}")

    except KeyboardInterrupt:
        pass

    return h_ctg, net.get_weights()


def render_greedy_policy(net, env, x0=None):
    '''Roll-out from random state using greedy policy.'''
    x0 = x = env.reset(x0)
    cost_to_go = 0.0
    gamma_i = 1
    for _ in range(MAX_EPISODE_LENGTH):
        # Create a batch made of a single state
        state_tensor = np2tf(np.array(x))

        # Get qvalues and select the index of the minimum for each joint
        q_values = net(state_tensor)        
        u_index = np.argmin(tf2np(q_values))
        u = []
        for i in reversed(range(env.nj)):
            u.append(u_index // (env.nu ** i))
            u_index -= env.nu ** i * u[env.nj - 1 - i]
        u.reverse()
        u = np.array(u)

        x, cost = env.step(u)
        cost_to_go += gamma_i * cost
        gamma_i *= DISCOUNT
        env.render()

    print(f"Real cost to go of state {x0} : {cost_to_go}")


def evalutate_greedy_policy(net, env, n_step=8):
    '''Evalute net using greedy policy on some predefined states.'''
    x = np.array([env.reset(np.vstack([np.array([[i*2*np.pi/n_step-np.pi] for _ in range(env.nq)]), np.array([[0] for _ in range(env.nv)])])) for i in range(n_step)])
    # x = np.append(x, np.array([env.reset(np.vstack([np.array([[i*2*np.pi/n_step-np.pi] for _ in range(env.nq)]), np.array([[0.5] for _ in range(env.nv)])])) for i in range(n_step)]), axis=0)
    # x = np.append(x, np.array([env.reset(np.vstack([np.array([[i*2*np.pi/n_step-np.pi] for _ in range(env.nq)]), np.array([[-0.5] for _ in range(env.nv)])])) for i in range(n_step)]), axis=0)

    evaluated_cost_to_go = 0.0
    gamma_i = 1
    for _ in range(MAX_EPISODE_LENGTH):
        # Create a batch of all states
        state_tensor = tf.transpose(tf.squeeze(np2tf(x), axis=0), [1, 0, 2])

        # Get qvalues and select the index of the minimum for each joint
        q_values = net(state_tensor)        
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
        gamma_i *= DISCOUNT

    return evaluated_cost_to_go


if __name__=='__main__':
    ### Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


    ### Hyper paramaters
    NEPISODES = 10000                       # Number of training episodes
    NPRINT = 1000                           # print something every NPRINT episodes
    MAX_EPISODE_LENGTH = 100       # TODO: we should reset it to 200?         # Max episode length
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
    nu = 11     # number of discretization steps for the joint torque u
    nj = 2      # number of joints of the pendolum

    env = Pendulum(nbJoint=nj, nu=nu)

    # Create critic and target NNs
    Q = get_critic(env.nx, env.nu, env.nj)
    Q_target = get_critic(env.nx, env.nu, env.nj)

    # Load NN weights from file
    #Q.load_weights("qbestweights1.h5")

    # Set initial weights of targets equal to those of actor and critic
    Q_target.set_weights(Q.get_weights())

    # Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    # Trains the network
    h_ctg, best_weights = train(Q, Q_target, agent)  # return the average cost to go to plot it and the best weights

    # Save NN weights to file (in HDF5)
    Q.set_weights(best_weights)
    Q.save_weights("qbestweights1.h5")

    render_greedy_policy(Q, env)

    print("Average cost: %.3f" % (sum(h_ctg)/len(h_ctg)))
    plt.plot( np.cumsum(h_ctg)/range(1, len(h_ctg) + 1) )
    plt.title("Average cost-to-go")
    plt.show()
