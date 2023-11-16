import torch
import torch.nn as nn

class MomentumCorrector(nn.Module):
    def __init__(self, learning_rate, rate_m):
        super().__init__()

        self.m = 0.0
        self.learning_rate = learning_rate
        self.rate_m = rate_m

    def forward(self, grad):
        self.m = self.rate_m * self.m + grad
        gradient = self.learning_rate * self.m

        return gradient
    
class AdamCorrector(nn.Module):
    def __init__(self, learning_rate, rate_m, rate_v, num_diff_timestemps):
        super().__init__()

        self.m = 0.0
        self.v = 0.0
        self.learning_rate = learning_rate
        self.rate_m = rate_m
        self.rate_v = rate_v
        self.num_diffusion_timesteps = num_diff_timestemps
        
    def forward(self, grad, t):
        self.m = self.m * self.rate_m + (1 - self.rate_m) * grad
        self.v = self.v * self.rate_v + (1 - self.rate_v) * grad ** 2

        m_hat = self.m / (1 - self.rate_m ** (self.num_diffusion_timesteps - t + 1))
        v_hat = self.v / (1 - self.rate_v ** (self.num_diffusion_timesteps - t + 1))

        grad = self.learning_rate * m_hat / (v_hat.sqrt() + 1e-8)

        return grad
