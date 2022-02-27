# Budget Optimisation
Budget optimisation is library to help you allocate your investement in different marketing channels in order to maximize a KPI of your interest, like revenue. 

# Usage 
pip install git+https://github.com/facundodeza/budget_optimisation.git

# budget_optimisation

After that, we can generate fake data using:
from budget_optimisation import *
df_test = make_dataframe(2, 1)
The first argument of the function is how many channels you want. The second argument is how many markets you want

If we plot them we will that none of them shows a perfect curve but we can see that grows at the beginning and then start to get flatter. 
You might find something like this in real life and the main issue is the great variability. This means with a similar level of marketing spend (spend column), you might get very different levels of your KPI (like revenue).

# How do you deal with this? 

In Lift [3], they get the distribution of a and b parameters and use them for sampling curves from them. Using these sampled curves, they decide. Later in this tutorial, we will evaluate how convenient it is.
We use bootstrapping for getting the distribution. It is important to say that a and b are normally distributed.

dict_market = get_distribution(df_test, 1000, 1000, 'market_column', 'channel_column', 'spend_column', 'kpi_column', None)

For example, Channel 1 and its parameter a is normally distributed with a mean of 775.6970 and a standard deviation of 108.2592. For its b parameter, is normally distributed with a mean of 0.6731 and a standard deviation of 0.0134. 

# Why do we simulate?
The main reason is the only thing that we can control is how much we invest in each channel. We do not know how the curves are when we make our decision or how they will be during our implementation. That is the reason we need the distributions for getting different possible curves based on our data and therefore, be able to test our decision process about how we allocate budget.

For example, for channel 1 we might have:
The orange curve might be the time where you have invested 42500 and get 750 thousand in revenue and the green curve is when you get 1.2 million in revenue with the same investment. 
The same thing is for channel 2

# How do we simulate and evaluate?
1) We got a sampled curve for each channel, imagine we got the green curve for channel 1 and the orange curve for channel 2. 
2) We optimized based on a given budget (for example, the budget we have spent in December 2021). The optimisation gives us how much to invest in channel 1 and channel 2.
3) We got another sampled curve for each channel. In this case, we get the orange curve for channel 1 and the green curve for channel 2.
4) Using the curves in 3), we get the expected revenue (KPI) investing in each channel based on 2). 
5) Using the curves in 3), we get the expected revenue (KPI) investing in each channel based on how the budget was allocated in December 2021 between channel 1 and channel 2.
6) We compare if 4) is bigger than 5) and its variation.
7) Repeat this process N times. 

Following our example:
In our data, we will run this process 5000 times for each month of 2021.

checking_dates = ['21-01', '21-02', '21-03', '21-04', '21-05', '21-06', '21-07', '21-08', '21-09', '21-10', '21-11', '21-12']
markets = ['Market 1']
summary, g_summary = simulation_summary(checking_dates, markets, df_test, 5000, dict_market, 'market_column', 'channel_column', 'spend_column', 'kpi_column', 'date_column')

We get :
In January 2021, 63.4% of times 4) was bigger than 5) but this is not reflected having a revenue increment. 
In April 2021, 98.3% of times 4) was bigger than 5) and getting a revenue increment of 1.8%. This have would mean an extra revenue of 17385.65 (kpi_variation multiply by observed_kpi)
We can do a similar analysis for the rest of the months.

In summary, we have that:

For the whole year, on average, 71.97% of the times 4) was bigger than 5). The net revenue (KPI) increment was 47657.88.
The decision for every month

We have seen that using simulation is convenient for assessing our new decision process. Now, if we want to decide for a given month, we can use:

from datetime import datetime

We fix a seed for month and year. Therefore, the user will not repeat until he gets something he likes. The sampled curves must be random. 

seed = datetime.today().month * datetime.today().year

# Set the budget you want to spend for a given month.
budget_to_be_spent = 50000

# Get sample curve
sim_helper_decision =  random_a_b_values(1, ['Market 1'], df_test['channel_column'].unique(), seed, dict_market) 
params_decision = get_params_vectors(sim_helper_decision)

# Optimize
optimized_budget = params_decision.apply(lambda x : budget_allocation(x['a_s'], x['b_s'], budget_to_be_spent), axis = 1 )
print(budget_allocation)

# Conclusions
To be able to optimize, first, we need to set a good attribution model. This plays a key role because is which define how much KPI is attributed for each channel. 
Using non-linear programming help us to optimize an objective function that is not linear and get how much to invest in each channel. 
Simulation helps us to assess in a world full of uncertainty if the decision we got from non-linear programming is better than our actual way to allocate resources.

# References
[1] https://adequate.digital/en/web-analytics/attribution-modeling/lancuchy-markowa-modelowanie-atrybucji-w-praktyce-cz-11
[2] https://medium.com/analytics-vidhya/the-shapley-value-approach-to-multi-touch-attribution-marketing-model-e345b35f3359
[3] https://eng.lyft.com/lyft-marketing-automation-b43b7b7537cc
