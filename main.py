import numpy as np
from utils import *
import argparse


def TSDE(cost,transition_matrix,num_runs,horizon):
    """
    This code computes the regret of the TSDE algorithm for tabular RL
    cost - cost matrix (shape m,n)
    transition matrix - (shape m,n,n)
    """

    m,n = cost.shape 
    initial_alpha = 0.1*np.ones((m,n,n))
    J_star,_,_ = policy_iteration(cost,transition_matrix)
    print('Optimal cost',J_star)
    regret = np.zeros(shape=(num_runs,horizon))

    for i in range(num_runs):

        print('Starting iteration {}'.format(i))
        # Initial state and parameters of dirichelet posterior
        x = np.random.randint(n)
        #x = 0
        alpha = initial_alpha.copy()
        
        
        t_last = 0      # t_last - Last sampling time
        T_last = 0      # T_last - Last episode length

        rho = np.zeros(shape=(n,m)) # State-action pair visits counter
        rho_base = np.zeros(shape=(n,m))

        total_regret = 0
        
        for t in range(horizon):

            if t-t_last >= T_last + 1 or np.any(rho>2*rho_base) or t==0:

                theta = drchrnd(alpha)
                _,_,g = policy_iteration(cost,theta)
                
                T_last = t-t_last
                t_last = t
                rho_base = rho
            
            u = g[x]
            total_regret += cost[u,x]-J_star

            # Updating alpha and generating next state
            x_next = gen_markov_state(transition_matrix[u,x,:])
            alpha[u,x,x_next] += 1
            rho[x,u] += 1

            x = x_next
            regret[i,t] = total_regret
        


    return regret




def main():
    
    n_state = 6
    n_control = 2 
    
    cost = np.random.random(size=(n_control,n_state))
    transition_matrix = np.zeros(shape=(n_control,n_state,n_state))

    # Riverswim problem
    """
    cost = np.array([[0.8,1,1,1,1,1],[1,1,1,1,1,0]])
    transition_matrix[0,:,:] = np.array([[1.0,0,0,0,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]])
    transition_matrix[1,:,:] = np.array([[0.7,0.3,0,0,0,0],[0.1,0.6,0.3,0,0,0],[0,0.1,0.6,0.3,0,0],[0,0,0.1,0.6,0.3,0],[0,0,0,0.1,0.6,0.3],[0,0,0,0,0.7,0.3]])

    """
    for u in range(n_control):
        for x in range(n_state):
            transition_matrix[u,x,:] = np.random.dirichlet(0.1*np.ones(n_state))

    
    regret = TSDE(cost,transition_matrix,num_runs=100,horizon=10000)
    print('Plotting the regret')
    plot_regret(regret[:,::100])


if __name__ == "__main__":
    
    riverswim_cost = np.array([[0.8,1,1,1,1,1],[1,1,1,1,1,0]])
    
    riverswim_transition = np.zeros((2,6,6))
    riverswim_transition[0,:,:] = np.array([[1.0,0,0,0,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]])
    riverswim_transition[1,:,:] = np.array([[0.7,0.3,0,0,0,0],[0.1,0.6,0.3,0,0,0],[0,0.1,0.6,0.3,0,0],[0,0,0.1,0.6,0.3,0],[0,0,0,0.1,0.6,0.3],[0,0,0,0,0.7,0.3]])

    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-T','--horizon',default=50000,type=int,help='Time Horizon')
    parser.add_argument('-c','--cost',default=riverswim_cost,help='Cost function')
    parser.add_argument('-P','--trans_prob',default=riverswim_transition,help='Probability Transition Matrix')
    parser.add_argument('-N','--num_runs',default=300,type=int,help='Number of rollouts for computing average regret')
    parser.add_argument('-rand','--rand_mdp',action='store_true',help='Random MDP')
    parser.add_argument('-n_s','--num_states',default=6,type=int,help='Number of states')
    parser.add_argument('-n_a','--num_actions',default=3,type=int,help='Number of actions')

    args = vars(parser.parse_args())

    if args['rand_mdp']:
        
        cost = np.random.random(size=(args['num_actions'],args['num_states']))
        transition_matrix = np.zeros(shape=(args['num_actions'],args['num_states'],args['num_states']))
        for u in range(args['num_actions']):
            for x in range(args['num_states']):
                transition_matrix[u,x,:] = np.random.dirichlet(0.1*np.ones(args['num_states']))
    else:

        cost = args['cost']
        transition_matrix = args['trans_prob']

        
    regret = TSDE(cost,transition_matrix,num_runs=args['num_runs'],horizon=args['horizon'])
    print('Plotting regret')
    plot_regret(regret[:,::100])











