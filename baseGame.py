import gym

class Actor:
    def __init__(self):
        pass

    def action(self, observation):
        pass ## IMPLEMENT ME! ##

    def on_game_end(self, game):
        pass ## IMPLEMENT ME! ##


class Game:
    def __init__(self, env=gym.make('CartPole-v1')):
        self.env = env

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        return self.env.reset()

    def run(self, actor, render=True, max_steps=10000):
        observation = self.reset()
        for i in range(max_steps):
            action = actor.action(observation)
            old_observation = observation
            observation, reward, done, info = self. env.step(action)
            self.recordStep(old_observation, action, reward)
            if render:
                self.env.render()
            if done:
                actor.on_game_end(self)
                return

    def recordStep(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
