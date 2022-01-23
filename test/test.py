
import sys
sys.path.append('../')

from budget_optimisation.budget_optimisation import  *
import pytest


def test_budget_allocation():
    
    """
    Usage: 
        We test that the constraint optimisation is being met. All budget must be spent.
    """
    
    a_s = [800.02, 9808.32, 350.22]
    b_s = [0.64, 0.35, 0.72]
    budget_to_be_spent = 25300
    
    
    optimized_budget = budget_allocation(a_s, b_s, budget_to_be_spent)
        
    assert round(np.sum(optimized_budget)) == budget_to_be_spent



def test_get_distribution():
    
    """
    Usage: 
        We test that we get the correct dictionary size in all levels.
    """    
    
    df_test = make_dataframe(2, 1)
    
    n_channel = len(df_test['channel_column'].unique())
    n_market = len(df_test['market_column'].unique())
    
    channel_a = df_test['channel_column'].unique()[0]
    market_a = df_test['market_column'].unique()[0]
    
    dict_market = get_distribution(df_test, 10, 10, 'market_column', 'channel_column', 'spend_column', 'kpi_column', None)
    
    assert (n_channel * n_market) == len(dict_market)
    assert len(dict_market[market_a +'_'+ channel_a]) == 2
    assert len(dict_market[market_a +'_'+ channel_a][0]['a']) == 2
    assert len(dict_market[market_a +'_'+ channel_a][1]['b']) == 2


def test_random_a_b_values():
    
    """
    Usage: 
        We make sure we get sim_helper dimensions right.
    
    """
    n_sim = 100
    
    df_test = make_dataframe(4, 2)
    channels = df_test['channel_column'].unique()
    markets = df_test['market_column'].unique()
    dict_market = get_distribution(df_test, 10, 10, 'market_column', 'channel_column', 'spend_column', 'kpi_column', None)
    
    sim_helper = random_a_b_values(n_sim , markets, channels, 0, dict_market)
        
    assert len(sim_helper) == n_sim * len(channels) * len(markets)
    assert sim_helper.reset_index()['index'].nunique() == n_sim


def test_get_params_vectors():
    
    """
    Usage:
        We make sure that our params dimensions are correct. 
    """

    n_sim = 100
    
    df_test = make_dataframe(3, 2)
    channels = df_test['channel_column'].unique()
    markets = df_test['market_column'].unique()
    dict_market = get_distribution(df_test, 10, 10, 'market_column', 'channel_column', 'spend_column', 'kpi_column', None)
    
    sim_helper = random_a_b_values(n_sim , markets, channels, 0, dict_market)
    
    params = get_params_vectors(sim_helper)
    
    assert len(params) == n_sim
    assert len(params['a_s'].iloc[0]) == (len(channels) * len(markets))
    assert len(params['a_s'].iloc[0]) == len(params['b_s'].iloc[0])



def test_quantify_uncertainty_simulations():
    
    """
    Usage:
        We make sure that our return values are float and success rate is between 0 and 1.
    """
    
    n_sim = 100
    
    df_test = make_dataframe(3, 2)
    channels = df_test['channel_column'].unique()
    markets = df_test['market_column'].unique()
    dict_market = get_distribution(df_test, 10, 10, 'market_column', 'channel_column', 'spend_column', 'kpi_column', None)
      
    success_rate, rate_variation_kpi, real_variation_kpi_unit, _ = quantify_uncertainty_simulations(df_test, 100, 'Market 1', dict_market, '20-07', 'market_column', 'channel_column', 'spend_column', 'kpi_column', 'date_column')

    assert success_rate <= 1
    assert success_rate >= 0
    assert isinstance(success_rate, float)
    assert isinstance(rate_variation_kpi, float)
    assert isinstance(real_variation_kpi_unit, float)


def test_make_dataframe():
    
    """
    Usage:
        We make sure that our fake data is in good shape.
    """    
    n_channels = 3
    n_markets = 2
    
    df_test = make_dataframe(n_channels, n_markets)
    
    df = ge.dataset.PandasDataset(df_test)
    expected_columns = ['date_column', 'channel_column', 'market_column', 'spend_column', 'kpi_column']
    result = df.expect_table_columns_to_match_ordered_list(column_list=expected_columns)
    
    assert result['success'] == True
    assert len(df_test) == (n_channels * n_markets * len(df_test['date_column'].unique()))





