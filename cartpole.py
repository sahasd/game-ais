import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    next_state, reward, done,_ = env.step(env.action_space.sample())
    print(next_state, reward, done,_)
