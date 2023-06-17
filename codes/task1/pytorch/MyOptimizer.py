import torch

class BaseOptimizer():
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class GdOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad

class AdamOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001,b1=0.9, b2=0.999,epsilon=1e-8):
        super().__init__(params, lr)
        self.beta1 = b1
        self.beta2 = b2
        self.epsilon = epsilon
        self.momentums = [torch.zeros_like(param) for param in self.params]
        self.velocities = [torch.zeros_like(param) for param in self.params]
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                # 动量更新
                self.momentums[i] = self.beta1 * self.momentums[i] + (1 - self.beta1) * param.grad
                # 更新参数更新的方向与大小（速度）
                self.velocities[i] = self.beta2 * self.velocities[i] + (1 - self.beta2) * param.grad**2
                param.data = param.data -  (self.lr/ (self.velocities[i].sqrt()+ self.epsilon)) * \
                		self.momentums[i]