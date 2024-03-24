import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


#q(xt | xt-1) = N(xt, sqrt(1-b)*xt-1, b)




def cosine_schedule(x_t):
    """
        Takes image of time step t as input and adds noise using cosine scheduler
    """

    return



def timestep_encoding(x_t):

    return


def linear_schedule(n_steps, start=0.0001, end=0.02):
    """
        Returns all the beta values indexed by their corresponding timestep
    """
    return torch.linspace(start, end, n_steps)


#Can be fixed for time step T
n_steps = 1000
t = 200
betas = linear_schedule(n_steps) 
alphas = 1 - betas
alpha_t_cumprod = torch.cumprod(alphas) #cumulative product of the alpha values for all time steps

def q_step(x_zero, timestep):
    """
        Forward diffusion step - uses sampling during closed form expression to return the sample for timestep t 
        Args:
            x_zero: input image at time step 0
            timestep: time step to sample from
        Return:
            noisy image at time step t
    """
    sqrt_alpha_t_cumprod = torch.sqrt(alpha_t_cumprod[timestep]) #square root of cumulative product of alphas up to time step t
    sampled_noise = torch.sqrt(1 - alpha_t_cumprod[timestep]) * torch.rand_like(x_zero) #scaled random numbers from standard normal distribution (z-distribution)
    x_t = x_zero * sqrt_alpha_t_cumprod + sampled_noise  
    return x_t


def p_step(x_t):
    """
        Reverse diffusion step
    """


    return 