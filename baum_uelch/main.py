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
np.random.seed(12)
count=100
n=100
I=[1,2,3,4]
K=[0,1]
random_P=False
random_Q=False

P_t = pd.read_excel('./Ptrue.xlsx')
P_t.head()
P_t = np.array(P_t.to_numpy(), float)
if random_P:
    P_r = np.random.randint(10,90,(2,2))
    #for i in range(len(P_r)):
        #P_r[i][i]+=10
    P_r=P_t*P_r
    for i in range(len(P_r)):
        sum_p=np.sum(P_r[i])
        P_t[i]= [j/sum_p for j in P_r[i]]

Q_t = pd.read_excel('./Qtrue.xlsx')
Q_t.head()
Q_t=np.array(Q_t.to_numpy(),float)
if random_Q:
    Q_r=np.random.randint(10,90,(2,4))
    #for i in range(len(Q_r)):
     #   Q_r[i][i] += 10
    Q_r=Q_t*Q_r
    for i in range(len(Q_r)):
        sum_p=np.sum(Q_r[i])
        Q_t[i]= [j/sum_p for j in Q_r[i]]
X=np.zeros((n,), dtype=int)
X[0]=1
for i in range(1,n):
    X[i]=random.choices([0,1],weights=P_t[X[i-1]],k=1)[0]
print(X)
s=len(Q_t)
ksi=len(Q_t[0])
n=len(X)

sigma=np.zeros((n,), dtype=int)
for i in range(n):
    sigma[i]=random.choices(I,weights=Q_t[X[i]] ,k=1)[0]
print(sigma)
P_ta = pd.read_excel('./Ptable.xlsx')
P_ta.head()
P=np.array(P_ta.to_numpy(),float)
Q_tab = pd.read_excel('./Qtable.xlsx')
Q_tab.head()
Q=np.array(Q_tab.to_numpy(),float)
stop=False
lambd=[0.5,0.5]

N=len(lambd)
P_0=P
Q_0=Q
lambd_0=lambd
eps=0.0001
u_to_prit2=[]
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

    df = pd.DataFrame(lambd_2)
    df.to_excel('newlambda2.xlsx', index=False)
    df = pd.DataFrame(P_2)
    df.to_excel('newP2.xlsx', index=False)
    df = pd.DataFrame(Q_2)
    df.to_excel('newQ2.xlsx', index=False)

    u_to_prit2.append(np.sum([np.sum([(alpha[i][m] * beta[i][m]) for m in range(n)]) for i in range(s)]))
    print(str(it+1) + ")", u_to_prit2[it + 1])
    if((u_to_prit2[it+1]-u_to_prit2[it])/u_to_prit2[it+1]<eps and not stop):
        print('!stop')
        count=it+4
        stop=True
    else: it+=1

plt.figure(1)
x = np.arange (0, count+2)
plt.title('Функция максимального правдоподобия', fontsize=14, fontname='Times New Roman')
plt.plot(x, u_to_prit2, ':o', label=r'$u(\widehat{{\sigma}^{*}},\widehat{{Z}^{*}})$')
plt.legend()

v=1
max_u=0
prog_s=[0]*n



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

    # Path Array
    S = np.zeros(T)

    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])

    S[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)

    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append(0)
        else:
            result.append(1)

    return result

plt.figure(2)
max_u=0
prog_s2=viterbi(sigma,P_2,Q_2,lambd_2)
x = np.linspace(0, n,n)

model = hmm.MultinomialHMM(n_components=2, n_iter=n)
XX=np.array([[i] for i in sigma])

plt.plot(x, X, '-o', label='X')
plt.plot(x, prog_s2, ':o', label=r'$\widehat{{X}^{*}}$')
plt.yticks( [i for i in range(2)])
plt.legend()

discrepancy_true_and_prognos=0
for i in range(n):
    if X[i]==prog_s2[i]:discrepancy_true_and_prognos+=1
discrepancy_true_and_prognos/=n
print('совпадение прогноза с изначальной последовательностью',discrepancy_true_and_prognos*100,"%")
print(prog_s2)

fig, ax = plt.subplots()

plt.title('Q', fontsize=14, fontname='Times New Roman')
im=ax.imshow(Q_t, cmap=plt.get_cmap('viridis', n),vmax=1,vmin=0)
#plt.colorbar([i for i in np.arange(0,1.5,0.25)] )
#cbar = plt.colorbar(ax.pcolor(Q_to_print))
plt.colorbar(im, ax=ax)

fig, ax = plt.subplots()
plt.title('P', fontsize=14, fontname='Times New Roman')
im=ax.imshow(P_t, cmap=plt.get_cmap('viridis', n),vmax=1,vmin=0)
plt.colorbar(im, ax=ax)
fig, ax = plt.subplots()
plt.pcolor([X,sigma,prog_s2], cmap=plt.get_cmap('viridis', n))

plt.yticks([i for i in np.arange(0,3.5,0.5)],[" ","X"," ","σ"," ",r'$\widehat{{X}^{*}}$'," "])
cbar = plt.colorbar(ax.pcolor([X,sigma,prog_s2]), ticks=[0,1, 2, 3, 4])
cbar.ax.set_yticklabels(['0','1', '2', '3', '4'])

plt.show()