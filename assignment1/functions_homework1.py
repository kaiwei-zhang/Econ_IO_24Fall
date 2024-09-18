from configure import *

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


def share_function_MNL(params, demand_df, temp_t,temp_price):
    """This is a mapping: given params&demand_df(I need sugar,caffeine, Diet,Regular),
    period t, and len-10 vector price, generate expected market share which is also len-10 vector"""
    [alpha,beta1,beta2,gammaD,gammaR] = params
    temp_demand_df = demand_df[demand_df['t']==temp_t]
    temp_demand_df['M'] = temp_demand_df['sugar']*beta1+temp_demand_df['caffeine']*beta2+temp_demand_df['Diet']*gammaD+temp_demand_df['Regular']*gammaR
    temp_demand_df['exp_M']=np.exp(temp_demand_df['M'])
    temp_demand_df['Price']=temp_price
    temp_demand_df['middle']=temp_demand_df['Price']*alpha
    temp_demand_df['exp_P'] = np.exp(temp_demand_df['middle'])
    temp_demand_df['exp_delta']=temp_demand_df['exp_M']*temp_demand_df['exp_P']
    denom=temp_demand_df['exp_delta'].sum()
    temp_demand_df['expected_share'] = temp_demand_df['exp_delta']/denom

    temp_demand_df = temp_demand_df.sort_index() 
    # to make sure the sort is correct so I can check j's expected share by [j-1]
    return list(temp_demand_df['expected_share'])