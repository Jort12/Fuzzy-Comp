from attr import Out
import torch
import torch.nn as nn
import torch.optim as optim
import itertools

"""
Gaussian Membership Function Layer:
    Purpose: To compute the degree of membership of inputs to fuzzy sets using Gaussian functions.
Sugeno FIS Layer: tweak the sigma and the center values

    

"""
class GaussianMF(nn.Module):
    
    def __init__(self,input_name, num_mfs,sigmas):
        super().__init__()# No parameters to initialize
        self.input_name = input_name #which input this MF layer is for
        self.num_mfs = num_mfs#number of membership functions
        self.sigmas = torch.tensor(sigmas) #list of sigmas for each MF, example: [0.3, 0.5, 0.7], so how wide each MF is
        self.log_sigmas = nn.Parameter(torch.log(self.sigmas)) #log sigmas to be learned
        self.centers = nn.Parameter(torch.randn(num_mfs))#centers of the MFs to be learned
    
    def forward(self,x):
        memberships = []
        for i in range(self.num_mfs):
            c = self.centers[i]
            s = self.sigmas[i]
            mu = torch.exp(-0.5 * ((x - c) / s) ** 2)
            memberships.append(mu)
        memberships = torch.stack(memberships, dim=1)

        return memberships


class RuleLayer(nn.Module):
    def __init__ (self, num_input, num_mfs):
        super().__init__()
        self.num_input = num_input
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** num_input #total number of rules

        #Rule combinations -- itertools.product: creates a cartesian product of input MFs
        self.rule_indices = list(itertools.product(range(num_mfs), repeat=num_input)) #all possible combinations of MFs for the inputs
    
    def forward(self, mf_outputs):
        #mf_outputs: list of tensors, one per input, each of shape (batch_size, num_mfs)
        batch_size = mf_outputs[0].shape[0]
        rule_strengths = []

        for i in self.rule_indices:
            weight = torch.ones(batch_size, device=mf_outputs[0].device)
            
            for j, input_name in enumerate(i):
                weight*= mf_outputs[j][:, input_name] #get the membership degree for the j-th input and the corresponding MF index

            rule_strengths.append(weight)

        return torch.stack(rule_strengths, dim=1) #(batch_size, num_rules)