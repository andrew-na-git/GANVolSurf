from libraries import *

# define cross entropy for discriminator loss function
class BCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(BCELoss, self).__init__()
        
    def forward(self, predictions, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction='sum')
        return bce

# define RMSE and soft constraints for generator loss function based on ackerer 2020
class RMSENoArbitrageLoss(nn.Module):
    def __init__(self, eps=1e-6, penalty=None):
        super(RMSENoArbitrageLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.penalty = penalty
        self.eps = eps
        
    def forward(self, input, predictions, targets, dloss):
        rmse = torch.sqrt(self.loss(predictions, targets) + self.eps)
        # arbitrage soft constraints
        m_idx = 0 # moneyness
        T_idx = 3 # time to maturity
        m = input[:, m_idx]
        T = input[:, T_idx]
        sigma = predictions.squeeze(1)
        w = torch.pow(sigma,2)*T

        # Gradients, note that predictions is implied volatility, and total variance (w) is predictions^2 * T
        # by chain rule and product rule, we get the following
        first_gradients = torch.autograd.grad(w.unsqueeze(1), input, grad_outputs=torch.ones_like(predictions), create_graph=True, retain_graph=True)[0]
        dwdt = first_gradients[:, T_idx]
        dwdm = first_gradients[:, m_idx]
        second_gradients = torch.autograd.grad(dwdm.unsqueeze(1), input, grad_outputs=torch.ones_like(predictions), retain_graph=True)[0]
        d2wdm2 = second_gradients[:, m_idx]
        # Calendar arbitrage (Monotonicity of T)
        c4_loss = torch.nn.functional.relu(torch.neg(dwdt))
        ## Butterfly arbitrage (Durrleman's cond)
        butterfly_helper = torch.pow(1 - m*dwdm/2*w, 2) - dwdm/4 * (1/w + 1/4) + d2wdm2/2
        c5_loss = torch.nn.functional.relu(torch.neg(butterfly_helper))
        ## Large-moneyness behavior
        c6_loss = torch.abs(d2wdm2)
        total_loss = self.penalty[0]*rmse + self.penalty[1]*c4_loss.mean() + self.penalty[2]*c5_loss.mean() + self.penalty[3]*c6_loss.mean() + self.penalty[4]*dloss

        return total_loss, c4_loss, c5_loss