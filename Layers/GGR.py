import torch as pt
import torch.nn as nn
from Functions.ExpR import ExpR
from Functions.LorR import LorR
from Functions.PlinR import PlinR


# Dictionary for van der waals radii for tau initialization
VanDerWaal = {'H':1.2, 'C':1.7, 'N':1.55, 'O':1.52, 'F':1.8, 'P':1.75, 'S':1.85, 'Cl':1.47, 'Br':1.8, 'I':1.98}

# Apply the functions you will be using
ExpR = ExpR.apply
LorR = LorR.apply
PlinR = PlinR.apply

# Create dictionary for easy use
function = {'exp':ExpR, 'lor':LorR, 'plin':PlinR}


class ESGGR(nn.Module):  # Element Specific Geometric Graph Representation Layer

    def __init__(self, eta_rng=None, kappa_rng=(5, 8), ker='lor', tau_rng=(0.5, 1.5), elems1=VanDerWaal.keys(), elems2=VanDerWaal.keys()):

        # Allow methods from nn.Module
        super(ESGGR, self).__init__()

        # Set initial options as attributes of the layer
        self.eta_rng = eta_rng
        self.kappa_rng = kappa_rng
        self.ker = ker
        self.tau_rng = tau_rng

        # The number of element-specific groups and number of features created fro output
        self.num_grps = len(elems1) * len(elems2)
        self.num_feat = 4 * self.num_grps

        # Uniformly initialize  kappa values
        self.kappa = nn.Parameter(pt.DoubleTensor(self.num_grps).uniform_(kappa_rng[0], kappa_rng[1]), requires_grad=True)

        # If a tau range was chosen, then use the Van Der Waal radius to calculate, otherwise use the eta range
        if not tau_rng:
            self.eta = nn.Parameter(pt.DoubleTensor(self.num_grps).uniform_(eta_rng[0], eta_rng[1]), requires_grad=True)
        else:
            tau = pt.DoubleTensor(1).uniform_(tau_rng[0], tau_rng[1])
            eta = pt.DoubleTensor(pt.zeros(self.num_grps))
            for i, elem1 in enumerate(elems1):
                for j, elem2 in enumerate(elems2):
                    eta[i*10+j] = VanDerWaal[elem1] + VanDerWaal[elem2]
            self.eta = nn.Parameter(tau * eta, requires_grad=True)

    def reset_parameters(self):
        if tau_rng:
            self.eta = nn.Parameter(pt.DoubleTensor(self.num_grps).uniform_(self.tau_rng[0], self.tau_rng[1]), requires_grad=True)
        else:
            self.eta = nn.Parameter(pt.DoubleTensor(self.num_grps).uniform_(self.eta_rng[0], self.eta_rng[1]), requires_grad=True)
        self.kappa = nn.Parameter(pt.DoubleTensor(self.num_grps).uniform_(self.kappa_rng[0], self.kappa_rng[1]), requires_grad=True)

    def forward(self, input):
        for param in self.parameters():  # The kappa and eta are clamped to avoid non-sensical calculations
            param.data.clamp_(min=0.01)
        return function[self.ker](input, self.eta, self.kappa)
