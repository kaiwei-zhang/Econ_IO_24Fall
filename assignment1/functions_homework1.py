from configure import *
from scipy.optimize import minimize


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



def run_2sls(df,Y_lis,X_lis, IV_lis,endog_list):
    # X stands for all RHS in OLS
    # IV_lis should be IV in 2sls
    # endog_lis is the list for endog variables
    # so for 2sls rhs = X+IV_lis-endog_lis
    # run_2sls(data,['log_odds'],['price', 'sugar', 'caffeine', 'Diet', 'Regular'],[],[]) would be OLS
    
    from statsmodels.sandbox.regression.gmm import IV2SLS
    X = df[X_lis]
    endog = df[endog_list]
    # IV_lis+endog_lis-X_lis
    exog_list = IV_lis+X_lis
    exog_list = [item for item in exog_list if item not in endog_list]
    IV = df[exog_list]
    model = IV2SLS(df[Y_lis], X, IV)
    return model.fit()
