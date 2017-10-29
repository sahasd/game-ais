import gym
import numpy as np
import tensorflow as tf
from gym import wrappers 

#Deep Q-learning algorithm
# 1. Do a feedforward pass for the current state s to get predicted Q-values for all actions.
# 2. Do a feedforward pass for the next state s′ and calculate maximum over all network outputs maxa′Q(s′,a′).
# 3. Set Q-value target for action a to r+γmaxa′Q(s′,a′) (use the max calculated in step 2). For all other actions, set the Q-value target to the same as originally returned from step 1, making the error 0 for those outputs.
# 4. Update the weights using backpropagation.


# creates the frozenlake environment
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/frozenlake-qlearning', force=True)


#neural network
x = tf.placeholder(shape=[1, 4], dtype=tf.float32)
y_ = tf.placeholder(shape=[1, 2], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([4, 2], 0, 0.1))
y = tf.matmul(x, W)
action = tf.argmax(y, 1)

cost = tf.reduce_sum(tf.square(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


#tensorflow initialization
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

num_episodes = 10000
episode = 0
gamma = 0.99
noise_factor = 0.1
state = env.reset()
turn = 0
while episode < num_episodes:
	turn = turn + 1
	act, qvals = sess.run([action, y], {x: [state]})

	if np.random.rand(1) < noise_factor:
		act[0] = env.action_space.sample()

	next_state, reward, done,_ = env.step(act[0])

	#calculate target Q value
	nqvals = sess.run([y], {x: [next_state]})
	maxQ = np.max(nqvals)
	targetQ = qvals
	# Q(s,a) = r + gamma * Q'(s',a')
	targetQ[0,[act[0]]] = reward + gamma * maxQ
	sess.run([optimizer], feed_dict={x: [next_state], y_: targetQ})

	state = next_state

	if done:
		episode = episode + 1
		state = env.reset()
		print("finished episode %d in %d turns and got reward %d" %(episode, turn, reward))
		turn = 0
		noise_factor = 1/((episode/100) + 10)













