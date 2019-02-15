from tensorNetwork import TensorActor

class QActor(TensorActor):
    def __init__(self):
        # super().__init__() # Python 3
        super(QActor, self).__init__() # Python 2
        self.max_memory_size = 300
        self.discount_factor = 0.5

    def reward_function(self, game, game_i, step_i, max_steps):
        # q = r + γQ∗(s', a')
        try:
            next_observation = game.observations[step_i+1]
            _, max_q = self.score_actions(next_observation)
            r = self.current_reward(game, game_i, step_i, max_steps)
            return r + self.discount_factor * max_q
        except IndexError as e: #at the end
            return 0

    def current_reward(self, game, game_i, step_i, max_steps):
        raise Exception('IMPLEMENT')

    def score_actions(self, observation):
        max_q = None
        max_action = None
        for action in self.get_discrete_actions():
            q = self.fire(observation, action)[0]
            if max_q is None or q > max_q:
                max_q = q
                max_action = action
        return max_action, max_q

    def get_best_action(self, observation):
        max_action, _ = self.score_actions(observation)
        return max_action
