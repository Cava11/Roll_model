"""
Roll Model
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

"Parameter setting"

T=20000 #dimension
c=0.01  #noninformational costs per trades
m0=50
su=0.01 #standar deviation

"Model simulation"

q=2*np.random.binomial(1, 0.5, size=T) - 1  #vector of random binomial numers - 1

"np.random.normal(0,su,T) generates an array of T random numbers from the normal distribution with mean 0 and standard deviation su"
"cumsum(x) takes the cumulative sum oft these numbers"
m=m0+np.cumsum(np.random.normal(loc=0, scale=su, size=T))
p=m+q*c # Price incorporating costs (including transaction costs)

"SIGNATURE PLOT"
Lmax=50
tau = np.arange(1, Lmax + 1)  # Time lags
# Precompute the signature
sig = np.zeros(Lmax)
for tau_val in tau:
    # Calculate the difference for higher orders manually (lag tau)
    diff_p = p[tau_val:] - p[:-tau_val]  # Equivalent to np.diff(p, n=tau_val) but handled manually
    sig[tau_val - 1] = np.mean(diff_p ** 2) / tau_val  # Compute signature

c_tau = su**2 + 2 * c**2 / tau  # Theoretical signature curve, based on Roll model assumptions

# Plot the results
plt.plot(tau, c_tau, label='Theoretical curve')  # Theoretical curve
plt.plot(tau, sig, 'o', label='Simulated points')  # Simulated signature points
plt.axhline(y=su**2, color='r', label="su^2")  # Line for su^2 (constant volatility term)

# Plot labels and legend
plt.xlabel('Tau')
plt.ylabel('C(tau)')
plt.title('Signature Plot (Roll Model)')
plt.legend()
plt.show()
# %%
"Autocorrelation function"

plot_acf(acf((np.diff(p))))
#Legend
plt.xlabel('diff(p)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function plot')
plt.legend()
plt.show()
# %%

'Estimating the spread. Notice that the value of sqrt(-gamma1) is very close to c=00.1 above (and the spread is 2c).'
# Calcolare l'autocorrelazione con il tipo "covariance"
# In statsmodels, 'acf' calcola l'autocorrelazione per default (corrisponde a 'acf' di R)
diff_p = np.diff(p)
a = acf(diff_p, fft=True, nlags=len(diff_p)-1)

# Accesso ai primi tre valori di 'acf'
acf_values = a[0:3]

# Calcolare la radice quadrata del negativo del secondo elemento
sqrt_value = np.sqrt(-a[1])

plt.plot(p, color='blue')
plt.plot(m, color='red', label='Index')  # Equivalent to col=2 in R
plt.xlabel('Index')
plt.ylabel('p')
plt.title('Estimating the spread')
plt.legend()
plt.show()


# %%
'Estimating the spread. Notice that the value of sqrt(-gamma1) is very close to c=00.1 above (and the spread is 2c).'
plt.plot(p[:20], 'o', label='Simulated point')
plt.plot(m[:20], color='red')
plt.xlabel('Index')
plt.ylabel('p[1:20]')
plt.legend()
plt.show()

#%%

'Estimating the filtered state estimate'

diff_p = np.diff(p)
#Creare il modello ARIMA(0,0,1) per diff_p
arima_model = ARIMA(diff_p, order=(0,0,1))
fitted_model = arima_model.fit()

# print(fitted_model.summary())
#Ottenere i coefficenti e i residui
coef = fitted_model.params
residuals = fitted_model.resid

#Calcolare i valori previsti f
f = p[1:] + coef[0]*residuals

plt.plot(p[:20],'o', label='p [1:20]')
plt.plot(m[:20], color='red', label='m [1:20]')
plt.plot(np.arange(1, 20), f[:19], color='green', label='f [1:19]')
plt.xlabel('Index')
plt.ylabel('p [1:20]')
plt.legend()
plt.show()