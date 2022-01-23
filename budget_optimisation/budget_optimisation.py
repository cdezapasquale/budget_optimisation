
import pandas as pd
import numpy as np
import pickle
import os
import random
import great_expectations as ge
from scipy.optimize import minimize, curve_fit


def curve(x : float, a : float, b : float) -> float: # pragma: no cover, basic curve
    
    """
    Usage: 
        The power function used in this package.
    
    Args:
        a: parameter value. An unique value or array is admitted.
        x: marketing spend. An unique value or array is admitted.
        b: parameter value. An unique value or array is admitted.
        
    Return:
        A float number or an array which is/are the estimated value/s of our variable of interest.
    
    """

    return np.multiply(a, np.power(x, b))





def make_dataframe(n_channels : int, n_markets: int) -> pd.core.frame.DataFrame:
    
    """
    Usage:
        For generating fake data and testing the functions
        
    Args:
        n_channels: Number of channels
        n_markets: Number of markets
    
    Return:
        A dataframe with 5 columns, having date_column, channel_columns, market_columns, spend_column and kpi_column
    """
    
    if n_channels < 2:
        raise ValueError('Number of channels must be greater than 1')
    
    min_investing_to_be_spent = 1e4
    max_investing_to_be_spent = 5e4
    
    dates = ['19-01', '19-02', '19-03', '19-04', '19-05', '19-06', '19-07', '19-08', '19-09', '19-10', '19-11', '19-12',
            '20-01', '20-02', '20-03', '20-04', '20-05', '20-06', '20-07', '20-08', '20-09', '20-10', '20-11', '20-12',
            '21-01', '21-02', '21-03', '21-04', '21-05', '21-06', '21-07', '21-08', '21-09', '21-10', '21-11', '21-12']

    df = pd.DataFrame()
    
    for i in range(1, n_channels + 1): 
        for j in range(1, n_markets + 1):
            for n, date in enumerate(dates):
                
                k = round(random.uniform(min_investing_to_be_spent, max_investing_to_be_spent ), 2)
               
                tmp = pd.DataFrame()
                                                
                tmp['date_column'] = [date]
                tmp['channel_column'] = ['Channel ' + str(i)]
                tmp['market_column'] = ['Market ' + str(j)]
                tmp['spend_column'] = [k]
                
                random.seed(i * 15 + n + j)
                a = random.uniform(400, 450)
                b = random.uniform(0.7, 0.76) 
                
                tmp['kpi_column'] = [round(curve(k, a, b), 2)]
       
                df = pd.concat([df, tmp], axis = 0)        
    
    return df.reset_index(drop = True)





def get_distribution(df : pd.core.frame.DataFrame, n_sim : int, boot_size : int, market_column : str, channel_column : str, spend_column : str, kpi_column : str, dict_path : str) -> pd.core.frame.DataFrame:
    
    """
    Usage:
        This function get the normal distribution for each a and b values for each market and channel using bootstrapping.
        
    Args: 
        df: Dataframe which have five columns. Date column, market column, channel column, spending column and kpi column.
        n_sim: Number of bootstrapping to be performed. 
        boot_size: Bootstrap size for each bootstrapping.
        market_column: The column name where we have the different market´s names.
        channel_column: The column name where we have the different channel´s names.
        spend_column: The column name where we have how much was invested.
        kpi_column: The column name which want to maximize, for example: revenue, LTV.
        dict_path: The path where the dict with a and b distributions will be saved. 
        
    Return:
        A dict with a and b distributions (mean and std) for each channel and market. 

    """
          
    dict_market = dict()
    
    df = df.reset_index()
        
    for j in df[market_column].unique() :
        for k in df[channel_column].unique():
            
            a = []
            b = []
                        
            df_tmp = df[(df[market_column] == j) & (df[channel_column] == k)].reset_index(drop = True)
            
            for i in range(n_sim):
        
                np.random.seed(i)
                el_index = np.array(np.random.choice(len(df_tmp), boot_size))  
                tmp_opt = df_tmp.iloc[el_index].reset_index(drop=True)
                popt, _ = curve_fit(curve, tmp_opt[spend_column].to_numpy(), tmp_opt[kpi_column].to_numpy())
                a.append(popt[0])
                b.append(popt[1])

            a_mean = np.mean(a)
            a_std = np.std(a)   

            b_mean = np.mean(b)
            b_std = np.std(b)   

            a_dict = dict({'a': [a_mean, a_std] })
            b_dict = dict({'b': [b_mean, b_std] })
    
            dict_market[j+'_'+k] = [a_dict, b_dict]
         
        
    if isinstance(dict_path, str) :
        
        # Create dir if it does not exist
        if os.path.isdir(dict_path) == False:
            os.mkdir(dict_path)
    
        f = open(dict_path,"wb")

        # write the python object (dict) to pickle file
        pickle.dump(dict_market, f)

        # close file
        f.close()
     
    return dict_market





def random_a_b_values(n_sim : int, markets : list, channels : list, seed : int, dict_path : str) -> pd.core.frame.DataFrame:

    """
    Usage: 
        This function simulates n_sim of "a" and "b" parameters values for optimisation and evaluation.
    
    Args:
        n_sim: Number of simulations to be performed.
        markets: A list of different markets that your business operate.
        channels: A list of channels where you want to optimize the budget allocation.
        seed: A int value used for generation pseudo-random values. Assure reproducibility. 
        dict_path: The path where the dict having the a and b distribution parameters for each market and channel.
        
    Return:
        sim_helper: A dataframe having n_sim of a and b values for a each channel. 
        
    """
    if isinstance(dict_path, str):
        with open(dict_path, "rb") as tf:
            dict_market = pickle.load(tf)
     
    elif isinstance(dict_path, dict):
        dict_market = dict_path
            
            
    sim_helper = pd.DataFrame()

    for i in markets:
        for j in channels:
            
            a_mean = dict_market[str(i+'_'+j)][0]['a'][0]
            a_std = dict_market[str(i+'_'+j)][0]['a'][1]

            b_mean = dict_market[str(i+'_'+j)][1]['b'][0]
            b_std = dict_market[str(i+'_'+j)][1]['b'][1]
            
            np.random.seed(seed)
            a_values = np.random.normal(loc = a_mean , scale = a_std, size = n_sim)  
            b_values = np.random.normal(loc = b_mean , scale = b_std, size = n_sim)  
            
            tmp = pd.DataFrame()
                        
            tmp['a_values'] = a_values
            tmp['b_values'] = b_values
            
            tmp['market'] = i
            tmp['channel'] = j
            
            sim_helper = pd.concat([sim_helper, tmp], axis = 0)
                       
    return sim_helper





def budget_allocation(a_s : list, b_s : list, budget_to_be_spent : float) -> np.ndarray:
    
    """
    Usage: 
        This function will allocate how much to invest in each channel for maximizing our KPI.
        
    Args:
        a_s: A vector of a parameters, one per channel. At least 2 values must have.
        b_s: A vector of b parametres, one per channel. At least 2 values must have.
        budget_to_be_spent: total budget to be spent in different channels.
        
    Return:
        Vector with the amount to invest in each channel. 
        
    """
    # Function to be maximized
    def objective(x):
        return -1 * np.sum(curve(x, a_s, b_s))
    
    # Restriction. All the budget must be spent.
    def constraint1(x):
        return np.sum(x) - budget_to_be_spent 

     
    # initial guesses. Each channel start with 0 budget. 
    n = len(a_s)
    x0 = np.zeros(n)
    
    for i in range(n):
        x0[i] = 0

    # optimize
    b = (1, budget_to_be_spent) 
    bnds = (b, ) * n
    con1 = {'type': 'eq', 'fun': constraint1}
    cons = ([con1])
    solution = minimize(objective, x0,method='SLSQP',                  bounds=bnds,constraints=cons)
    
    x = solution.x
    
    return x





def get_params_vectors(sim_helper : pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    
    """
    Usage: 
        It regroups a and b values for each simulation. Each simulation is given by "Index"
    
    Args:
        sim_helper: a dataframe which have a and b values for each channel and market.
    
    Return:
        a dataframe with the regrouped a and b values for each simulation.
    
    """
    
    a_s = sim_helper.reset_index().sort_values(by=['index', 'channel']).groupby(['index'])['a_values'].apply(list)
    b_s = sim_helper.reset_index().sort_values(by=['index', 'channel']).groupby(['index'])['b_values'].apply(list)

    params = pd.DataFrame()
    params['a_s'] = a_s
    params['b_s'] = b_s

    
    return params





def data_check(df : pd.core.frame.DataFrame, market_column : str, channel_column : str, spend_column : str, kpi_column : str, 
                                     date_column : str) -> str: # pragma: no cover, using great expectations
    
    """ 
    Usage:
        Checking that data is correct (at least in some aspects)
    """
    
    # Check that our required columns exist
    
    df_ge = ge.dataset.PandasDataset(df)
    
    expected_columns = [date_column, channel_column, market_column, spend_column, kpi_column]
    result = df_ge.expect_table_columns_to_match_set(column_set  = expected_columns)  
        
    if result['success'] == False:
        raise ValueError('You have missed a column. Remember that you need a date_column, channel_column, market_column, spend_column and KPI column')
        
    # Checking for nulls
        
    s = df_ge.expect_column_values_to_not_be_null(column = market_column)
    
    if s['success'] == False:
        raise ValueError('You have null values in your market column. It should be fixed.')
    
    s = df_ge.expect_column_values_to_not_be_null(column = channel_column)
    
    if s['success'] == False:
        raise ValueError('You have null values in your channel column. It should be fixed.')    
        
    s = df_ge.expect_column_values_to_not_be_null(column = spend_column)
    
    if s['success'] == False:
        raise ValueError('You have null values in your spend column. It should be fixed.') 
        
    s = df_ge.expect_column_values_to_not_be_null(column = kpi_column )
    
    if s['success'] == False:
        raise ValueError('You have null values in your KPI column. It should be fixed.') 
        
    # Check for number of channels   
    if len(df[channel_column].unique()) < 2:
        raise ValueError("You need at least 2 channels to optimize.")
        





def channel_market_curve_plotting(df : pd.core.frame.DataFrame, spend_column : str, channel_column : str, 
                                  market_column : str, kpi_column, curves : bool = False, dict_path : str = None):
    
    """
    Usage:
        This function will plot a scatter plot where x-axis is spend_column and y-axis is kpi_column.
        You can plot alternative curves as well. Optional.
        
    Args: 
        df: It is dataframe which have five columns. date column, market column, channel column, spend column and kpi column.
        spend_column: The column name where we have how much was invested.
        channel_column: The column name where we have the different channel´s names.
        market_column: The column name where we have the different market´s names.
        kpi_column: The column name which want to maximize, for example: revenue, LTV.
        curves: If we want to plot alternative fitted curves along with the observed data.
        dict_path: The path where the dict having the a and b distribution parameters for each market and channel, will be loaded.

    """    
    
    for channel in df[channel_column].unique():
        for market in df[market_column].unique():
            
            tmp = df[(df[channel_column] == channel) & (df[market_column] == market)]          
            
            if curves:
                if dict_path is None:
                    raise ValueError('You need to provide a distribution dict for plotting altenartive curves')          
                    
                sim_helper = random_a_b_values(20, [market], [channel], 0, dict_path) 

                tmp['Alt Curve 1'] = tmp[spend_column].apply(lambda x: curve(x, sim_helper['a_values'].iloc[0] , sim_helper['b_values'].iloc[0]))
                tmp['Alt Curve 2'] = tmp[spend_column].apply(lambda x: curve(x, sim_helper['a_values'].iloc[1] , sim_helper['b_values'].iloc[1]))
                
                tmp = tmp[[spend_column,  kpi_column, 'Alt Curve 1',  'Alt Curve 2']].set_index(spend_column)
                
                tmp.columns = ['Observed data', 'Alt Curve 1', 'Alt Curve 2']
                
                tmp.plot(style='o', title = 'Observed data + alternative curves for ' + channel + ' in '+ market , figsize = (15, 10))
                
            else:
                tmp.plot.scatter(x = spend_column, y = kpi_column, title = channel + ' in '+ market, figsize = (15, 10))      





def quantify_uncertainty_simulations(df : pd.core.frame.DataFrame, n_sim : int, market : list, dict_path : str, date : str,  
                                     market_column : str, channel_column : str, spend_column : str, kpi_column : str, 
                                     date_column : str) -> tuple:
    
    """
    Usage:
        Evaluate how good is the new decision process comparing with the empirical one, for a given date and market.
        
    Args:
        df: It is dataframe which have five columns. date column, market column, channel column, spend column and kpi column.
        n_sim: Number of simulations to be performed. 
        market: A given market to filter df and we want evaluate our decision process.
        dict_path: The path where the dict having the a and b distribution parameters for each market and channel, will be loaded.
        date: A given date to filter df and we want evaluate our decision process. String with format 'YYYY-MM-DD', 'YY-MM.
        market_column: The column name where we have the different market´s names.
        channel_column: The column name where we have the different channel´s names.
        spend_column: The column name where we have how much was invested.
        kpi_column: The column name which want to maximize, for example: revenue, LTV.
        date_column: The column name where we have the different dates we might want to evaluate.
        
    Return:
        success_rate: % where optimized decision was bigger than the empirical decision.
        variation_rate: % of variation between optimized decision and empirical decision.
        variation_number: The extra KPI (like revenue) that we would have gotten if we had applied the optimized decision instead.
    """
                    
    data_check(df, market_column, channel_column, spend_column, kpi_column, date_column)
        
    # We use date as seed for generating random a and b values.
    date_n = [int(item.replace('-', '')) for item in [date]][0]

    # We filter the data
    total_budget = df[(df[date_column] == date) & (df[market_column] == market)].reset_index()
    
    # We get how much was spent in each channel. Empirical allocation
    actual_assig = total_budget.groupby([channel_column])[spend_column].sum().reset_index().sort_values(by = [channel_column])[spend_column].to_list()
    
    # We get the observed KPI (like revenue) for the given date and market.
    observed_y = total_budget[kpi_column].sum()
    
    if observed_y <= 0:
        raise ValueError("KPI must be bigger than 0")
          
    # For a given date and market, we get how much was spent in advertising.   
    budget_to_be_spent = total_budget[spend_column].sum()
        
    if budget_to_be_spent <= 0:
        raise ValueError("Budget must be bigger than 0")
        
    # Curves that will be used for optimisation.
    sim_helper_decision_uncertainty =  random_a_b_values(n_sim, [market], df[channel_column].unique(), date_n, dict_path) 
    params_decision_uncertainty = get_params_vectors(sim_helper_decision_uncertainty)

    # Curves that will be used for evaluation.
    sim_helper_reality_checking =  random_a_b_values(n_sim, [market], df[channel_column].unique(), date_n * 3, dict_path)
    params_reality_checking = get_params_vectors(sim_helper_reality_checking)    
    
    # We get the optimized allocation
    optimized_budget = params_decision_uncertainty.apply(lambda x : budget_allocation(x['a_s'], x['b_s'], budget_to_be_spent), axis = 1 )
    params_reality_checking['optimized_budget'] = optimized_budget
    
    # Expected KPI using optimized allocation.       
    optimized = params_reality_checking.apply(lambda x : np.sum(curve(x['optimized_budget'], x['a_s'], x['b_s'])), axis = 1)   
    
    # Expected KPI using empirical allocation.
    actual = params_reality_checking.apply(lambda x : np.sum(curve(actual_assig, x['a_s'], x['b_s'])), axis = 1) 
    
    # Calculations
    success_rate = round(np.sum(optimized > actual) / n_sim, 3)
    variation_rate = round(np.mean(np.divide(optimized, actual) - 1), 3)
    variation_number = round(observed_y * variation_rate, 2)
    
    return (success_rate, variation_rate, variation_number, observed_y)





def simulation_summary(checking_dates : list, markets : list, df : pd.core.frame.DataFrame, n_sim : int, dict_path : str, 
                                     market_column : str, channel_column : str, spend_column : str, kpi_column : str, 
                                     date_column : str) -> tuple: # pragma: no cover, pending
    
    """
    Usage:
        Get a summary of our simulations
        
    Args:
        checking_dates : a list of dates where you want to perform the simulation and get the results.
        markets : a list of markets where you want to perform the simulation and get the results.
        df: It is dataframe which have five columns. date column, market column, channel column, spend column and kpi column.
        n_sim: Number of simulations to be performed. 
        dict_path: The path where the dict having the a and b distribution parameters for each market and channel, will be loaded.
        market_column: The column name where we have the different market´s names.
        channel_column: The column name where we have the different channel´s names.
        spend_column: The column name where we have how much was invested.
        kpi_column: The column name which want to maximize, for example: revenue, LTV.
        date_column: The column name where we have the different dates we might want to evaluate.
        
    Return:
        results: dataframe with the result per month and market.
        global_summary : dataframe with the aggregated values for market.
    

    """
    succes_rate_ob_list = []
    variation_kpi_list = []
    real_variation_kpi_list = []
    observed_KPI_list = []
    market_list = []
    dates_list = []
    
    for date in checking_dates:
        for market in markets:

            success_rate_ob, variation_kpi, real_variation_kpi, observed_KPI = quantify_uncertainty_simulations(df, n_sim, market, dict_market, date, market_column, channel_column, spend_column, kpi_column, date_column)
            succes_rate_ob_list.append(success_rate_ob)
            variation_kpi_list.append(variation_kpi)
            real_variation_kpi_list.append(real_variation_kpi)
            market_list.append(market)
            dates_list.append(date)
            observed_KPI_list.append(observed_KPI)
            
    markets = pd.DataFrame(market_list, columns = ['markets'])
    dates = pd.DataFrame(dates_list, columns = ['dates'])
    succes_rate_ob_list = pd.DataFrame(succes_rate_ob_list, columns = ['success_rate'])
    variation_kpi_list = pd.DataFrame(variation_kpi_list, columns = ['KPI_variation'])
    real_variation_kpi_list = pd.DataFrame(real_variation_kpi_list, columns = ['real_variation_KPI']) 
    observed_KPI = pd.DataFrame(observed_KPI_list, columns = ['observed_KPI']) 

    results = pd.concat([dates, markets, succes_rate_ob_list, variation_kpi_list, real_variation_kpi_list, observed_KPI], axis = 1)
    
    global_summary = results.groupby('markets').agg({'success_rate': 'mean', 'real_variation_KPI': 'sum' , 'observed_KPI':'sum'}).reset_index()
    global_summary['global_variation'] = round(global_summary['real_variation_KPI'] / global_summary['observed_KPI'], 3)
    global_summary = global_summary[['markets', 'success_rate', 'real_variation_KPI', 'global_variation']]
   
    return results, global_summary
    







