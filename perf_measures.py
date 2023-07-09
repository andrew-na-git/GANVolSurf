from libraries import *

mean_abs_error = MeanAbsoluteError()
mean_abs_percentage_error = MeanAbsolutePercentageError()

def absolute_error(preds: Tensor, target: Tensor):
  return torch.abs(preds - target)

def absolute_percent_error(preds: Tensor, target: Tensor, epsilon: float = 1.17e-6):
  abs_diff = torch.abs(preds - target)
  abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)
  return abs_per_error