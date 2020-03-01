import numpy as np
import seaborn as sns
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt

def policy_iteration(c,P):
    """
    This function implements the policy iteration algorithm for infinite horizon MDPs
    with average cost criterion

    c - Cost Function (shape (|U|,|X|))
    P - Transition matrix (shape (|U|,|X|,|X|))

    """

    tolerance = 1e-3
    m,n = c.shape

    # Initial policy
    g = np.zeros(n).astype(int)

    x = np.arange(n)
    while True:

        # Solve the linear system of equations for the current policy (Policy evalutaion)
        P_g = P[g[x],x,:]
        A = np.eye(n) - P_g
        A = np.concatenate((np.ones(shape=(n,1)),A[:,1:]),axis=1)
        b = c[g[x],x]
        
        z = np.linalg.solve(A,b)
        #w = np.array([0,z[1:]])
        w = np.concatenate((np.zeros(1),z[1:]))
        #print(w.shape)

        # Policy improvement step
        temp = c + np.matmul(P,w)
        g = np.argmin(temp,axis=0)

        
        #Checking convergence
        M = z[0] + w
        X = np.min(temp,axis=0)
        if np.linalg.norm(M-X)<tolerance:
            break
    
    return z[0],w,g




def drchrnd(alpha):
    """
    Generates a sample from drichlet distribution with parameters alpha
    alpha - (|U|,|X|,|X|)
    """

    m,n,_ = alpha.shape
    P = np.zeros(shape=(m,n,n))

    for u in range(m):
        for x in range(n):
            P[u,x,:] = np.random.dirichlet(alpha[u,x,:])


    return P



def gen_markov_state(p):
    s = np.random.multinomial(1,pvals=p)
    return np.nonzero(s)[0][0]


def plot_regret(regret):
    
    df = pd.DataFrame(regret).melt()
    sns.lineplot(x='variable',y='value',data=df,label='Regret vs Time horizon')
    plt.xlabel('Horizon')
    plt.ylabel('Regret')
    plt.legend()
    plt.grid('on')
    plt.show()



