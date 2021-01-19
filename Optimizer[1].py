import numpy as np

class ScheduledOptim():

    def __init__(self, optimizer, learn_rate, d_mod, warmup):
        self.opt = optimizer
        self.learn_rate = learn_rate
        self.d_mod = d_mod
        self.warmup = warmup
        self.steps = 0

    def update(self):
        self.steps += 1
        lr = self.learn_rate * self.get_lr()
        for params in self.opt.param_groups:
            params['lr'] = lr
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


    def get_lr(self):
        d_mod = self.d_mod
        steps, warmup = self.steps, self.warmup
        return (d_mod ** -0.5) * min(steps ** (-0.5), steps * warmup ** (-1.5))



        

