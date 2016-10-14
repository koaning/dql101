import gym
import StateFitter as sf
import numpy as np

def test_valuemodel_creation():
    env = gym.make('CartPole-v0')
    vmod = sf.ValueModel(env)
    assert len(env.state) == vmod.input_size
    # pred = vmod.predict(np.random.normal(vmod.input_size))
    # assert len(env.action_space.n) == len(pred)

def test_valuemodel_creation2():
    env = gym.make('CartPole-v1')
    vmod = sf.ValueModel(env)
    assert len(env.state) == vmod.input_size
    # pred = vmod.predict([[np.random.normal(vmod.input_size)]])
    # print(pred)
    # assert len(env.action_space.n) == len(pred)
