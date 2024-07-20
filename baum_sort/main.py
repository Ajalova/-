import random

import pandas as pd
import numpy as np
import openpyxl
import math
import matplotlib.pyplot as plt
import matplotlib.dates
import random

random.seed(103)
np.random.seed(45)

#random.seed(106)
#random.seed(111)
#np.random.seed(45)
eps=0.0001
count=200

n=100
show_the_original_version=False
show_a_smoothed_version=True

change_Q=True
change_lambd=True

random_Q=False
random_P=False


sigma_tru=np.zeros((n,), dtype=int)
P_t=np.zeros((4,4), dtype=float)
Q_t=np.zeros((4,4), dtype=float)

P_t = pd.read_excel('./P_true.xlsx')
P_t.head()
P_t = np.array(P_t.to_numpy(), float)
if random_P:
    P_r = np.random.randint(10,90,(4,4))
    for i in range(len(P_r)):
        P_r[i][i]+=10
    P_r=P_t*P_r
    for i in range(len(P_r)):
        sum_p=np.sum(P_r[i])
        P_t[i]= [j/sum_p for j in P_r[i]]

sigma_tru[0]=1
for i in range(1,n):
    sigma_tru[i]=random.choices([1,2,3],weights=P_t[sigma_tru[i-1]-1],k=1)[0]
print(sigma_tru)
Q_t = pd.read_excel('./Q_true.xlsx')
Q_t.head()
Q_t=np.array(Q_t.to_numpy(),float)

if random_Q:
    Q_r=np.random.randint(10,90,(4,4))
    for i in range(len(Q_r)):
        Q_r[i][i] += 10
    Q_r=Q_t*Q_r
    for i in range(len(Q_r)):
        #Q_r[i][i]+=200
        sum_p=np.sum(Q_r[i])
        Q_t[i]= [j/sum_p for j in Q_r[i]]


s=len(Q_t)
ksi=len(Q_t[0])


sigma=np.zeros((n,), dtype=int)
for i in range(n):
    sigma[i]=random.choices([1,2,3],weights=Q_t[sigma_tru[i]-1],k=1)[0]


P_ta = pd.read_excel('./Ptable.xlsx')
P_ta.head()
P=np.array(P_ta.to_numpy(),float)
Q_tab = pd.read_excel('./Qtable.xlsx')
Q_tab.head()
Q=np.array(Q_tab.to_numpy(),float)
lambd=[0.5,0.25,0.25]

N=len(lambd)
t=100

X=np.zeros((t,n), dtype=float)

u_to_prit1=[]

lambd_s=np.zeros((s,), dtype=float)
P_s=np.zeros((s,s), dtype=float)
Q_s=np.zeros((s,ksi), dtype=float)
lambd_s = lambd.copy()
Q_s = Q.copy()
P_s = P.copy()

def Xgenerate(v,X,ul,i=0,prognoz=[0]*n,ans=[0]*n):
    if i==0:
        for j in range(1,N+1):
            v = lambd[j-1]
            prognoz[0]=j
            if v!=0: X,ul=Xgenerate(v,X,ul,1,prognoz,ans)
        return X,ul
    elif i<n:
        for j in range(1, N + 1):
            if (P[prognoz[i-1]-1][j-1]*Q[j-1][sigma[i] - 1])!=0:
                prognoz[i] = j
                X,ul=Xgenerate(v *( P[prognoz[i-1]-1][j-1]),X,ul, i+1, prognoz,ans)
        return X,ul
    else:
        if (len(ul)!=0):
            if (v < ul[-1] and len(ul) == 100):
                return X, ul
            k=len(ul)-1
            while(v > ul[k] and k!=0):
                k-=1
            ul.insert(k+1,v)
            X.insert(k+1, prognoz.copy())
            if(len(ul) == 101):
                ul.pop(100)
                X.pop(100)

        else:
            X.append(prognoz.copy())
            ul.append(v.copy())
        return X,ul

def u_tu_print(v,X,ul,lambd,P,Q,i=0,prognoz=[0]*n,ans=0):
    if i==0:
        for j in range(1,N+1):
            v = lambd[j-1] * Q[j-1][sigma[0] - 1]
            prognoz[0]=j
            if v!=0: ans=u_tu_print(v,X,ul,lambd,P,Q,1,prognoz,ans)
        return ans
    elif i<n:
        for j in range(1, N + 1):
            if ( P[prognoz[i-1]-1][j-1] * Q[j-1][sigma[i] - 1])!=0:
                prognoz[i] = j
                ans=u_tu_print(v *( P[prognoz[i-1]-1][j-1] * Q[j-1][sigma[i] - 1]),X,ul,lambd,P,Q, i+1, prognoz,ans)
        return ans
    else:
        ans += v
        return ans

u_to_prit1=[]
u_to_prit1.append(0)
u_to_prit2=[]
stop=False
u_to_prit2.append(0)
it=0
while (it<count):
    if show_the_original_version:
        X=[]
        ul=[]

        Xgenerate(0,X,ul,0,[0]*n)
        if(len(X)!=100):
            X = []
            ul = []
            Xgenerate(0,X,ul,0,[0]*n)
            print(len(X))
        t=len(X)
        #число переходов i→j в цепочке х(l)
        f = np.empty([t, s, s])
        r = np.zeros((t, s), dtype=float)
        nt = np.zeros((t, s, ksi), dtype=float)

        for l in range(t):
            for i in range(s):
                for j in range(0,s):
                    sum=0
                    for m in range(1, n):
                        if (it == 5):
                            pr = 8
                        if(X[l][m-1]==i+1 and X[l][m]==j+1):
                            sum+=1

                    f[l][i][j]=sum

                if(X[l][0]==i+1):  r[l][i]=1

                for j in range(0,ksi):
                    sum=0
                    for m in range(0,n):
                        if(X[l][m]==i+1 and sigma[m]==j+1): sum+=1
                    nt[l][i][j]=sum

        u=np.zeros((t,), dtype=float)
        for l in range(t):
            u[l]=lambd[X[l][0]-1]*Q[X[l][0]-1][sigma[0]-1]
            for i in range(1,n):
                u[l] *= P[X[l][i-1]-1][X[l][i]-1] * Q[X[l][i]-1][sigma[i]-1]


        e=np.zeros((s,), dtype=float)
        c=np.zeros((s,s), dtype=float)
        d=np.zeros((s,ksi), dtype=float)
        for i in range(s):
            for l in range(t):
                e[i]+=r[l][i]*u[l]
            for j in range(0, s):
                for l in range(t):
                    c[i][j] += f[l][i][j] * u[l]
            for k in range(0, ksi):
                for l in range(t):
                    d[i][k] += nt[l][i][k] * u[l]

        df = pd.DataFrame (e)
        df.to_excel('e.xlsx', index=False)
        df = pd.DataFrame (c)
        df.to_excel('c.xlsx', index=False)
        df = pd.DataFrame (d)
        df.to_excel('d.xlsx', index=False)

        lambd_1=np.zeros((s,), dtype=float)
        P_1=np.zeros((s,s), dtype=float)
        Q_1=np.zeros((s,ksi), dtype=float)

        moment=n//2
        s_e=np.sum(e)
        for i in range(0, s):
            lambd_1[i]=e[i]/s_e

            s_c=np.sum(c[i])
            if s_c==0:
                print('p')
            for j in range(0, s):
                P_1[i][j]=c[i][j]/s_c
            s_d=np.sum(d[i])
            if s_d == 0:
                print('g')
            for k in range(0, ksi):
                Q_1[i][k]=d[i][k]/s_d

        df = pd.DataFrame (lambd_1)
        df.to_excel('lambda.xlsx', index=False)
        df = pd.DataFrame (P_1)
        df.to_excel('newP.xlsx', index=False)
        df = pd.DataFrame (Q_1)
        df.to_excel('newQ.xlsx', index=False)

        P = P_1
        if change_Q: Q=Q_1
        if change_lambd: lambd=lambd_1

    if show_a_smoothed_version:

        alpha = np.zeros((s, n), dtype=float)
        beta = np.zeros((s, n), dtype=float)
        lambd_2 = np.zeros((s,), dtype=float)
        P_2 = np.zeros((s, s), dtype=float)
        Q_2 = np.zeros((s, ksi), dtype=float)

        for j in range(s):
            alpha[j][0] = lambd_s[j] * Q_s[j][sigma[0] - 1]
            beta[j][n - 1] = 1
        for m in range(1, n):
            for j in range(s):
                for i in range(0, s):
                    alpha[j][m] += alpha[i][m - 1] * P_s[i][j]
                alpha[j][m] *= Q_s[j][sigma[m] - 1]
        for m in range(n - 2, -1, -1):
            for j in range(s):
                for i in range(0, s):
                    beta[j][m] += beta[i][m + 1] * Q_s[i][sigma[m + 1] - 1] * P_s[j][i]
        C=np.sum([np.sum(alpha[i][0] * beta[i][0])  for i in range(s)])
        for i in range(0, s):
            lambd_2[i] = alpha[i][0] * beta[i][0] / C

            for j in range(0, s):
                s_p2 = 0
                s_p = 0
                for m in range(0, n - 1):
                    s_p += alpha[i][m] * beta[j][m + 1] * Q_s[j][sigma[ m + 1] - 1]  # /np.sum([np.sum([P[ii][j]*alpha[ii][m]*beta[jj][m+1]* Q[jj][sigma[m + 1] - 1] for ii in range(s)]) for jj in range(s)])
                    s_p2 += alpha[i][m] * beta[i][m]  # /(np.sum([alpha[ii][m]*beta[ii][m] for ii in range(s)]))
                P_2[i][j] = P_s[i][j] * (s_p / s_p2)
            for k in range(ksi):
                s_q = 0
                s_q2 = 0
                for m in range(0, n):
                    if (sigma[m] == k + 1):
                        s_q += alpha[i][m] * beta[i][m]  # /(np.sum([alpha[ii][m]*beta[ii][m] for ii in range(s)]))
                    s_q2 += alpha[i][m] * beta[i][m]  # /(np.sum([alpha[ii][m]*beta[ii][m] for ii in range(s)]))
                Q_2[i][k] = s_q / s_q2
        P_s = P_2
        if change_Q: Q_s = Q_2
        if change_lambd: lambd_s = lambd_2

        df = pd.DataFrame(lambd_2)
        df.to_excel('newlambda2.xlsx', index=False)
        df = pd.DataFrame(P_2)
        df.to_excel('newP2.xlsx', index=False)
        df = pd.DataFrame(Q_2)
        df.to_excel('newQ2.xlsx', index=False)



    if show_a_smoothed_version and not show_the_original_version:
        u_to_prit2.append(np.sum([np.sum([(alpha[i][m] * beta[i][m]) for m in range(n)]) for i in range(s)]))
        print(str(it + 1) + ")", u_to_prit2[it+1])
        if(u_to_prit2[it ]>0):
            if ((u_to_prit2[it ] - u_to_prit2[it-1]) / u_to_prit2[it] < eps and not stop):
                print('!stop')
                count = it + 4
                stop = True
            else:
                it += 1
        else:
            it += 1

    elif  show_the_original_version and not show_a_smoothed_version:
        u_to_prit1.append(u_tu_print(0,[],[],lambd,P,Q))
        print(str(it+1) + ")", u_to_prit1[it] )
        if (u_to_prit1[it] > 0):
            if ((u_to_prit1[it] - u_to_prit1[it - 1]) / u_to_prit1[it] < eps and not stop):
                print('!stop')
                count = it + 4
                stop = True
            else:
                it += 1
        else:
            it += 1
    elif show_a_smoothed_version and show_the_original_version:
        u_to_prit2.append(u_tu_print(0, [], [], lambd_2, P_2, Q_2))
        print(str(it + 1) + ")", u_to_prit2[it+1])
        u_to_prit1.append(u_tu_print(0, [], [], lambd, P, Q))
        print(str(it + 1) + ")", u_to_prit1[it+1])
        if (u_to_prit2[it] > 0 and u_to_prit1[it] > 0):
            if ((u_to_prit2[it] - u_to_prit2[it - 1]) / u_to_prit2[it] < eps and (u_to_prit1[it] - u_to_prit1[it - 1]) / u_to_prit1[it] < eps and not stop):
                print('!stop')
                count = it + 4
                stop = True
            else:
                it += 1
        else:
            it += 1
plt.figure()

plt.title('Функция максимального правдоподобия', fontsize=14, fontname='Times New Roman')
if show_the_original_version:
    x = np.arange(0, len(u_to_prit1))
    plt.plot(x, u_to_prit1, ':o', label=r'$u(\widehat{{\sigma}^{*}_{0}},\widehat{{Z}^{*}_{0}})$')
if show_a_smoothed_version:
    x = np.arange(0, len(u_to_prit2))
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
        result.append(s+1)
    return result


max_u=0

x = np.linspace(0, n-1, n)
if show_the_original_version:
    plt.figure()
    prog_s=viterbi(sigma, P, Q, lambd)
    plt.plot(x, sigma_tru, '-o', label='Х', linewidth=5, markersize=9)
    plt.plot(x, sigma, '-o', label='σ', linewidth=3, markersize=7, color='pink')
    plt.plot(x, prog_s, '-o', label=r'$\widehat{{X}^{*}_{0}}$', linewidth=1, markersize=5, color='red')
    plt.yticks([i for i in range(1, 4)])
    plt.legend()

if show_a_smoothed_version:
    plt.figure()
    prog_s2 = viterbi(sigma, P_s, Q_s, lambd_s)
    #plt.plot(x, sigma, '-o', label='sigma')
    plt.plot(x, sigma_tru, '-o', label='Х',linewidth= 5, markersize=9)
    plt.plot(x, sigma, '-o', label='σ', linewidth=3, markersize=7, color='pink')
    plt.plot(x, prog_s2, '-o', label=r'$\widehat{{X}^{*}}$', linewidth=1,markersize=5,color='red')
    plt.yticks( [i for i in range(1,4)])
    plt.legend()
    fig, ax = plt.subplots()
    plt.title('P', fontsize=14, fontname='Times New Roman')
    im = ax.imshow(P_t, cmap=plt.get_cmap('viridis', n), vmax=1, vmin=0)
    plt.colorbar(im, ax=ax)
    fig, ax = plt.subplots()
    plt.title('Q', fontsize=14, fontname='Times New Roman')
    im = ax.imshow(Q_t, cmap=plt.get_cmap('viridis', n), vmax=1, vmin=0)
    plt.colorbar(im, ax=ax)

    fig, ax = plt.subplots()
    plt.pcolor([sigma_tru,sigma,prog_s2], cmap=plt.get_cmap('viridis', n))

    plt.yticks([i for i in np.arange(0,3.5,0.5)],[" ","X"," ","σ"," ",r'$\widehat{{X}^{*}}$'," "])
    cbar = plt.colorbar(ax.pcolor([sigma_tru,sigma,prog_s2]), ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(['1', '2', '3', '4'])

    fig, ax = plt.subplots()
    plt.pcolor([sigma_tru, sigma], cmap=plt.get_cmap('viridis', n))

    plt.yticks([i for i in np.arange(0, 3.5, 0.5)], [" ", "X", " ", "σ", " ", r'$\widehat{{X}^{*}}$', " "])
    cbar = plt.colorbar(ax.pcolor([sigma_tru, sigma]), ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(['1', '2', '3', '4'])

print('изначальная последовательность',sigma_tru[0:10],"...",sigma_tru[-10:100])

print('искажонная последовательность',sigma[0:10],"...",sigma[-10:100])


if (show_a_smoothed_version):
    discrepancy_true_and_prognos2=0
    discrepancy_noise_and_prognos2=0
    for i in range(n):
        if sigma_tru[i] == prog_s2[i]: discrepancy_true_and_prognos2 += 1
        if sigma[i] == prog_s2[i]: discrepancy_noise_and_prognos2 += 1
    discrepancy_true_and_prognos2 /= n
    discrepancy_noise_and_prognos2 /= n
    print('прогноз2', np.array([int(i) for i in prog_s2]))
    print('совпадение прогноза2 с изначальной последовательностью', discrepancy_true_and_prognos2 * 100, "%")
    print("совпадение прогноза2 с зашумленной последовательностью", discrepancy_noise_and_prognos2 * 100, "%")
if show_the_original_version:
    discrepancy_true_and_prognos=0
    discrepancy_noise_and_prognos=0
    for i in range(n):
        if sigma_tru[i]==prog_s[i]:discrepancy_true_and_prognos+=1
        if sigma[i]==prog_s[i]:discrepancy_noise_and_prognos+=1
    discrepancy_true_and_prognos/=n
    discrepancy_noise_and_prognos/=n
    print('прогноз', [int(i) for i in prog_s])
    print('совпадение прогноза с изначальной последовательностью',discrepancy_true_and_prognos*100,"%")
    print("совпадение прогноза с зашумленной последовательностью",discrepancy_noise_and_prognos*100,"%")

if(random_Q):print(Q_t)
if(random_P):print(P_t)


plt.show()