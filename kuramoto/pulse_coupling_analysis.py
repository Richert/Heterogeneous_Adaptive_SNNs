import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
    return float(np.prod(np.arange(1, n + 1)))

def C_q(q, n):
    cq = 0.0
    for i in range(n+1):
        cq_tmp = 0.0
        for j in range(i+1):
            if i-2*j == q:
                cq_tmp += factorial(n)*(-1)**i/(2**i * factorial(n - i)*factorial(j)*factorial(i - j))
        cq += cq_tmp
    return cq

n = 2
for i in range(0, n+1):
    cq = C_q(i, n)
    print(rf"$C_{i} = {np.round(cq, decimals=2)}$")
a_n = 2**n * factorial(n)**2 / factorial(2*n)
print(fr"$a_{n} = {np.round(a_n, decimals=2)}$")