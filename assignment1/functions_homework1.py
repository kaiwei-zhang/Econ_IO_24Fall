from configure import *
from scipy.optimize import minimize

global demand_df,c_jt_dic,PARAMS,alpha, Temp_t, data

def prepare_data(data):
    # Calculate outside good share (assuming outside share is 1 - sum of inside shares)
    outside_share = 1 - data.groupby('t')['market_share'].sum()
    data = data.merge(outside_share.reset_index().rename(
        columns={'market_share': 'outside_share'}), on='t')

    # Calculate log odds ratio
    data['log_odds'] = np.log(data['market_share'] / data['outside_share'])

    # Create dummy variables for diet and regular sodas
    data['Diet'] = (data['nest'] == 'Diet').astype(int)
    data['Regular'] = (data['nest'] == 'Regular').astype(int)
    return data

# a function I use to summarize reg results
def get_parameters(model, nested=False, print_results=False):
    dic = {}
    alpha = model.params['price']
    beta1 = model.params['sugar']
    beta2 = model.params['caffeine']
    gamma_D = model.params['Diet']
    gamma_R = model.params['Regular']
    dic['alpha'] = alpha
    dic['beta1'] = beta1
    dic['beta2'] = beta2
    dic['gamma_D'] = gamma_D
    dic['gamma_R'] = gamma_R
    if nested:
        sigma = model.params['log_within_share']
        dic['sigma'] = sigma
    if print_results:
        print(f"alpha (price): {alpha:.4f}")
        print(f"beta1 (sugar): {beta1:.4f}")
        print(f"beta2 (caffeine): {beta2:.4f}")
        print(f"gamma_D (Diet): {gamma_D:.4f}")
        print(f"gamma_R (Regular): {gamma_R:.4f}")
        if nested:
            print(f"sigma : {sigma:.4f}")
    return dic


# a function I use for 3(c)
def paint(list1,list2,list3):
    # for x axis
    x = range(1, len(list1)+1)
    # figure 1 for 3 lists
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, list1, marker='o', label='case1: 10 independent firms')
    plt.plot(x, list2, marker='s', label='case2: 1 and 2 collude')
    plt.plot(x, list3, marker='^', label='case3: 10 collude')
    plt.xlabel('firm j')
    plt.ylabel('price')
    plt.title('Comparison of Three Price Lists')
    plt.legend()

    # figure 2 for diff
    plt.subplot(1, 2, 2)
    diff2 = np.array(list2) - np.array(list1)
    diff3 = np.array(list3) - np.array(list1)
    plt.plot(x, diff2, marker='s', label='Case2 - Case1')
    plt.plot(x, diff3, marker='^', label='Case3 - Case1')
    plt.xlabel('firm j')
    plt.ylabel('Difference')
    plt.title('Differences from List 1')
    plt.legend()
    plt.grid(True)

    # show
    plt.tight_layout()
    plt.show()



################### For Nash Solver Part###############
# a function for demand/share
def share_function_MNL(temp_price):
    global demand_df,c_jt_dic,PARAMS,alpha, Temp_t, data
    """This is a mapping: given params&demand_df(I need sugar,caffeine, Diet,Regular),
    period t, and len-10 vector price, generate expected market share which is also len-10 vector"""
    [alpha,beta1,beta2,gammaD,gammaR] = PARAMS
    temp_t=Temp_t
    temp_demand_df = demand_df[demand_df['t']==temp_t]
    temp_demand_df['M'] = temp_demand_df['sugar']*beta1+temp_demand_df['caffeine']*beta2+temp_demand_df['Diet']*gammaD+temp_demand_df['Regular']*gammaR
    temp_demand_df['exp_M']=np.exp(temp_demand_df['M'])
    temp_demand_df['Price']=temp_price
    temp_demand_df['middle']=temp_demand_df['Price']*alpha
    temp_demand_df['exp_P'] = np.exp(temp_demand_df['middle'])
    temp_demand_df['exp_delta']=temp_demand_df['exp_M']*temp_demand_df['exp_P']
    denom=temp_demand_df['exp_delta'].sum()+1 #+1 for outside option
    temp_demand_df['expected_share'] = temp_demand_df['exp_delta']/denom

    temp_demand_df = temp_demand_df.sort_index() 
    # to make sure the sort is correct so I can check j's expected share by [j-1]
    return list(temp_demand_df['expected_share'])

def solve_Nash(initial_guess,question=1):
    """Solve NE by minimizing norm of 10 equations. Return a 10-dim vector of prices"""
    global demand_df,c_jt_dic,PARAMS,alpha, Temp_t, data
    # a loss function for each j,t: given p, tell loss from j
    def loss(p_vec,j,s_list):
        p_j=p_vec[j-1]
        c_jt=c_jt_dic[(j,Temp_t)] #check the cost
        LHS= 1/(c_jt-p_j)
        s_jt=s_list[j-1] # the estimated market share under p_vec
        RHS=alpha*(1-s_jt)
        return abs(LHS-RHS)
        return (LHS-RHS)*(LHS-RHS)
    
    # another loss specifically for 3(b):merging between 1 and 2
    def loss_q3b(p_vec,s_list):
        p_1=p_vec[0]
        p_2=p_vec[1]

        c_1=c_jt_dic[(1,Temp_t)] #check the cost
        c_2=c_jt_dic[(2,Temp_t)] #check the cost

        s_1=s_list[0] # the estimated market share under p_vec
        s_2=s_list[1]
        def diff(p1,c1,s1,p2,c2,s2):
            lhs=1+(p1-c1)*alpha*(1-s1)
            rhs=(p2-c2)*alpha*s2
            return abs(lhs-rhs)

        return diff(p_1,c_1,s_1,p_2,c_2,s_2)+diff(p_2,c_2,s_2,p_1,c_1,s_1)

    

    # a Loss function: given p, tell loss from 10 firms
    def LOSS_Q_1(p_vec):
        s_list=share_function_MNL(p_vec)
        sum1=0
        for j in [1,2,3,4,5,6,7,8,9,10]:
            sum1+=loss(p_vec,j,s_list)
        return sum1
    
    def LOSS_Q_2(p_vec):
        s_list=share_function_MNL(p_vec)
        sum1=0
        for j in [3,4,5,6,7,8,9,10]:
            sum1+=loss(p_vec,j,s_list)
        sum1+=loss_q3b(p_vec,s_list)
        return sum1
    
    def LOSS_Q_3(p_vec):
        s_list=share_function_MNL(p_vec)
        p=p_vec
        sum1=0
        for j in [1,2,3,4,5,6,7,8,9,10]:
            lhs=1+(p[j-1]-c_jt_dic[(j,Temp_t)])*alpha*(1-s_list[j-1])
            rhs=0
            for k in [1,2,3,4,5,6,7,8,9,10]:
                if k!=j:
                    rhs+= (p[k-1]-c_jt_dic[(k,Temp_t)])*alpha*(s_list[k-1])
            sum1+= (lhs-rhs)**2
        return sum1


    if question ==1:
        result = minimize(LOSS_Q_1, initial_guess, tol=1e-5,method = 'Nelder-Mead')
    elif question ==2:
        result = minimize(LOSS_Q_2, initial_guess, tol=1e-5,method = 'Nelder-Mead')
    elif question ==3:
        result = minimize(LOSS_Q_3, initial_guess, tol=1e-8,method = 'Nelder-Mead')
        
    print('number of iterations for minimize kernal:', result.nit)
    price_star = result.x
    return price_star,result