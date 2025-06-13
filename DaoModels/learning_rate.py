import math


class LearningRateScheduler:
    def __init__(self, config):
        self.config = config
        self.learning_rate = config["learning_rate"]
        self.warmup_steps = config["warmup_steps"]
        self.total_steps = config["total_steps"]
        self.scheduler_type = config["learning_rate_scheduler"]

        if self.scheduler_type == "cosine":
            self.scheduler_function = self.cosine_annealing_lr
        elif self.scheduler_type == "linear":
            self.scheduler_function = self.linear_warmup_cosine_decay_lr
        elif self.scheduler_type == "three_steps":
            self.scheduler_function = self.three_steps_lr
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.scheduler_type}")

    def get_learning_rate(self, current_step):
        lr = self.scheduler_function(current_step)
        return lr


    def cosine_annealing_lr(self, current_step):
        if current_step < self.warmup_steps:
            return self.learning_rate * current_step / self.warmup_steps
        else:
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        
    def linear_warmup_cosine_decay_lr(self, current_step):
        if current_step < self.warmup_steps:
            return self.learning_rate * current_step / self.warmup_steps
        else:
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
    
    def three_steps_lr(self, current_step):
        if current_step < self.warmup_steps:
            return self.learning_rate * current_step / self.warmup_steps
        elif current_step < self.warmup_steps + self.total_steps:
            return self.learning_rate
        else:
            return self.learning_rate * 0.1
