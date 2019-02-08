from tensorNetwork import TensorActor

class QActor(TensorActor):
    def __init__(self):
        # super().__init__() # Python 3
        super(QActor, self).__init__() # Python 2
        # change settings here
        self.max_memory_size = 300
        self.round_batch_size = 50
        self.traning_epochs = 50

        self.discount_factor = 0.5

    def reward_function(self, game, game_i, step_i, max_steps):
        # current reward is survivability. final reward is death
        # q = r + γQ∗(s', a')
        try:
            next_state = game.observations[step_i+1]
            q_left = self.fire(next_state, 0)
            q_right = self.fire(next_state, 1)
            r = ((max_steps - step_i)/max_steps)
            return r + self.discount_factor * max(q_left, q_right)
        except IndexError as e: #at the end
            return 0

actor = QActor()
actor.run()
