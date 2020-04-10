import gym
import time
import yaml
import tensorflow as tf
import tensorflow.keras.layers as layers

from assignment2.dqn.agents import DQNAgent


def linear_model(env: gym.Env) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Dense(units=32, activation='relu', input_shape=env.observation_space.shape, kernel_initializer='he_uniform'),
        layers.Dense(units=32, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(units=env.action_space.n)
    ])
    return model

def conv_model(env: gym.Env) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Conv2D(filters=32, input_shape=(None, 84, 84, 4), kernel_size=8, strides=4, activation='relu'),
        layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'),
        layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(units=512, activation='relu'),
        layers.Dense(units=env.action_space.n, activation='linear')
    ])
    return model

def run_simulation(agent: DQNAgent, env: gym.Env, render: bool = True, n_times: int = 1) -> float:
    obs, done, ep_reward = env.reset(), False, 0
    for _ in range(n_times):
        while not done:
            action = agent.act(state=obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
                time.sleep(0.01)
        env.reset()
        done = False
    env.close()
    return ep_reward / n_times

def main():
    config = dict()

    with open('../configs/linear.yaml') as f:
        config = yaml.load(f)

    training_steps = int(config['training_steps'])
    gamma = float(config['gamma'])

    env = gym.make("CartPole-v0")

    model = linear_model(env)
    agent = DQNAgent(model, config)

    rewards_sum = run_simulation(agent, env, n_times=100, render=False)
    print(f'Before Training: {rewards_sum} out of 200')

    agent.learn(env, steps=training_steps, gamma=gamma)

    agent.model.save('linear')

    rewards_sum = run_simulation(agent, env, n_times=100, render=False)
    print(f'After Training: {rewards_sum} out of 200')

if __name__ == '__main__':
    main()
