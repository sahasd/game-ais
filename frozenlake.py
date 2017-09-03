import gym
import numpy as np
import tensorflow as tf
from gym import wrappers 

# creates the frozenlake environment
env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, '/tmp/frozenlake-qlearning', force=True)
n_obv = env.observation_space.n
n_acts = env.action_space.n


#neural network
x = tf.placeholder(shape=[1, 16], dtype=tf.float32)
y_ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.1))
y = tf.matmul(x, W)
action = tf.argmax(y, 1)

cost = tf.reduce_sum(tf.square(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


#tensorflow initialization
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

num_episodes = 10000
state = env.reset()

for episode in range(1):
	act, qvals = sess.run([action, y], {x: np.identity(16)[state:state + 1]})








