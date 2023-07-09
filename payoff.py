from libraries import *

"""
Parameters:
- S: stock price 
- K: strike price
- r: risk-free interest rate
- T: time to maturity (in year)
- sigma: volatility (implied)
"""

def european_call(S, K, r, T, sigma):
  """
  output vanilla European call option's price using Black-Scholes formula 
  """
  d1 = (np.log(S/K) + (r+sigma**2/2)*T) / (sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def european_put(S, K, r, T, sigma):
  """
  output vanilla European put option's price using Black-Scholes formula 
  """
  
  # use the put-call parity for the put price
  C = european_call(S, K, r, T, sigma)
  DK = K * np.exp(-r*T)
  return C + S - DK