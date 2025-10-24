import torch
import torch.nn as nn
import torch.optim as optim
import itertools

"""
Gaussian Membership Function Layer:
    Purpose: To compute the degree of membership of inputs to fuzzy sets using Gaussian functions.

RuleLayer:
    Purpose: To compute the firing strength of fuzzy rules based on the membership degrees from the MF layer.
Sugeno Layer: tweak the sigma and the center values

    

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
    

class SugenoLayer(nn.Module):
    def __init__(self, num_rules, num_inputs):
        super().__init__()
        self.num_rules = num_rules
        self.num_inputs = num_inputs

        #Consequent parameters: each rule has a linear function of inputs + bias
        self.consequents = nn.Parameter(torch.randn(num_rules, num_inputs + 1)) #+1 for bias term



    #rule_strengths: (batch_size, num_rules)
    #inputs: (batch_size, num_inputs)
    def forward(self, rule_strengths, inputs):
        batch_size = inputs.shape[0]

        #bias term to inputs
        inputs_with_bias = torch.cat([inputs, torch.ones(batch_size, 1, device=inputs.device)], dim=1) #(batch_size, num_inputs + 1)

        #Compute rule outputs. matmul returns the product of 2 tensors
        rule_outputs = torch.matmul(inputs_with_bias.unsqueeze(1), self.consequents.unsqueeze(0).transpose(1,2)).squeeze(1) #(batch_size, num_rules)

        #Weighted sum of rule outputs
        weighted_outputs = rule_strengths * rule_outputs #(batch_size, num_rules)
        output = torch.sum(weighted_outputs, dim=1) / (torch.sum(rule_strengths, dim=1) + 1e-6) #(batch_size,)

        return output
    

class SugenoNet(nn.Module):
    def __init__(self, num_inputs, num_mfs, num_outputs):
        super().__init__()
        #module lists for MF layers
        self.mf_layers = nn.ModuleList([
            GaussianMF(f"input_{i}", num_mfs, [0.3, 0.5, 0.7]) for i in range(num_inputs)
        ])
        self.rule_layer = RuleLayer(num_inputs, num_mfs)#number of rules = num_mfs^num_inputs
        self.sugeno_layer = SugenoLayer(num_rules=num_mfs ** num_inputs, num_inputs=num_inputs)

    def forward(self, x):
        mf_outputs = [mf(x[:, i]) for i, mf in enumerate(self.mf_layers)]#get MF outputs for each input
        rule_strengths = self.rule_layer(mf_outputs)#get rule strengths
        y = self.sugeno_layer(rule_strengths, x)#get final output
        return y.unsqueeze(1)  #(batch_size, 1)
