class OptimizerFactory:
    def __init__(self, config):
        self.config = config

    def get_optimizer(self):
        pass


class AdamOptimizer(OptimizerFactory):
    def __init__(self, config):
        super().__init__(config)