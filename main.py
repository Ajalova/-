import random
from hmmlearn import hmm
import pandas as pd
import numpy as np
import openpyxl
import math
import matplotlib.pyplot as plt
import matplotlib.dates
import random

random.seed(103)
np.random.seed(11)
count=200
I=[1,2,3,4]
K=[0,1,2,3]
random_P=False
random_Q=False
eps=0.0001
P_t = pd.read_excel('./P.xlsx')
P_t.head()
P_t = np.array(P_t.to_numpy(), float)

Q_t = pd.read_excel('./Q.xlsx')
Q_t.head()
Q_t=np.array(Q_t.to_numpy(),float)

X=np.array([3,	3,	2,	2,	1,	2,	2,	3,	2,	1,	3,	3])
s=len(Q_t)
ksi=len(Q_t[0])


sigma=np.array([4,3,3,4,4,4,4,4,4,4,4,4,4,1,3,4,3,2,2,2,2,2,3,2,1,4,3])
X=np.array(    [2,1,1,1,3,3,3,2,2,3,2,3,2,1,3,3,3,2,2,1,2,2,3,2,1,3,3])
P=P_t
Q=Q_t
n=len(sigma)
lambd=[0.34,0.33,0.33]

N=len(lambd)

u_to_prit2=[]
stop=False
u_to_prit2.append(0)
it=0
while (it<count):
    alpha = np.zeros((s, n), dtype=float)
    beta = np.zeros((s, n), dtype=float)
    lambd_2 = np.zeros((s,), dtype=float)
    P_2 = np.zeros((s, s), dtype=float)
    Q_2 = np.zeros((s, ksi), dtype=float)

    for j in range(s):
        alpha[j][0] = lambd[j] * Q[j][sigma[0] - 1]
        beta[j][n - 1] = 1
    for m in range(1, n):
        for j in range(s):
            for i in range(0, s):
                alpha[j][m] += alpha[i][m - 1] * P[i][j]
            alpha[j][m] *= Q[j][sigma[m] - 1]
    for m in range(n - 2, -1, -1):
        for j in range(s):
            for i in range(0, s):
                beta[j][m] += beta[i][m + 1] * Q[i][sigma[m + 1] - 1] * P[j][i]

    C=(np.sum([alpha[i][0] * beta[i][0] for i in range(s)]))
    for i in range(0, s):
        lambd_2[i] = alpha[i][0] * beta[i][0] / C

        for j in range(0, s):
            s_p2 = 0
            s_p = 0
            for m in range(0, n-1):
                s_p += alpha[i][m] * beta[j][m + 1] * Q[j][sigma[ m + 1] - 1]  # /np.sum([np.sum([P[ii][j]*alpha[ii][m]*beta[jj][m+1]* Q[jj][sigma[m + 1] - 1] for ii in range(s)]) for jj in range(s)])
                s_p2 += alpha[i][m] * beta[i][m]  # /(np.sum([alpha[ii][m]*beta[ii][m] for ii in range(s)]))
            #s_p2 += alpha[i][n-1] * beta[i][n-1]
            P_2[i][j] = P[i][j] * (s_p / s_p2)
        for k in range(ksi):
            s_q = 0
            s_q2 = 0
            for m in range(0, n):
                if (sigma[m] == k + 1):
                    s_q += alpha[i][m] * beta[i][m]  # /(np.sum([alpha[ii][m]*beta[ii][m] for ii in range(s)]))
                s_q2 += alpha[i][m] * beta[i][m]  # /(np.sum([alpha[ii][m]*beta[ii][m] for ii in range(s)]))
            Q_2[i][k] = s_q / s_q2
    P = P_2
    Q = Q_2
    lambd = lambd_2

    u_to_prit2.append(np.sum([np.sum([(alpha[i][m] * beta[i][m]) for m in range(n)]) for i in range(s)]))
    if (u_to_prit2[it] > 0):
        if ((u_to_prit2[it] - u_to_prit2[it - 1]) / u_to_prit2[it] < eps and not stop):
            print('!stop')
            count = it + 4
            stop = True
        else:
            it += 1
    else:
        it += 1
    print(str(it) + ")", u_to_prit2[it ])

df = pd.DataFrame(lambd_2)
df.to_excel('newlambda2.xlsx', index=False)
df = pd.DataFrame(P_2)
df.to_excel('newP2.xlsx', index=False)
df = pd.DataFrame(Q_2)
df.to_excel('newQ2.xlsx', index=False)



plt.figure(5)
x = np.arange (0, len(u_to_prit2))
plt.plot(x, u_to_prit2, ':o', label=r'$u(\widehat{{\sigma}^{*}},\widehat{{Z}^{*}})$')
#plt.grid(True)
plt.xticks([i for i in range(len(u_to_prit2))])
plt.legend()

v=1
max_u=0

def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]
    V=[i-1 for i in V]
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)

            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
    S = np.zeros(T)
    last_state = np.argmax(omega[T - 1, :])
    S[0] = last_state
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    S = np.flip(S, axis=0)
    result = []
    for s in S:
        result.append(s)

    return result

plt.figure(3)
max_u=0
prog_s=viterbi(sigma,P_2,Q_2,lambd_2)
prog_s=[i+1 for i in prog_s]
x = np.linspace(0, n-1,n)
plt.plot(x[-len(X)-1:], X, '-o', label='X')
plt.plot(x, prog_s, ':o', label=r'$\widehat{{X}^{*}}$')
x_ticks = [i for i in range(28)]
x_labels = [str(i) for i in range(1996,2024)]
plt.xticks(x_ticks,x_labels)
plt.yticks([i for i in range(1,4)],['А','Б','В'])
plt.legend()

discrepancy_true_and_prognos=0
delt=len(prog_s)-len(X)
for i in range(len(X)):
    if X[i]==prog_s[i+delt]:discrepancy_true_and_prognos+=1
discrepancy_true_and_prognos/=len(X)
print('совпадение прогноза с изначальной последовательностью',discrepancy_true_and_prognos*100,"%")
print(prog_s)
plt.figure()
x = np.linspace(0, 12,13)
plt.plot(x, X[-13:], '-o', label='X')
plt.plot(x, prog_s[-13:], ':o', label=r'$\widehat{{X}^{*}}$')
x_ticks = [i for i in range(13)]
x_labels = ["2011", "2012", "2013", "2014", "2015","2016", "2017", "2018", "2019", "2020","2021","2022","2023"]
plt.xticks(x_ticks,x_labels)
plt.yticks([i for i in range(1,4)],['А','Б','В'])
plt.legend()
plt.show()