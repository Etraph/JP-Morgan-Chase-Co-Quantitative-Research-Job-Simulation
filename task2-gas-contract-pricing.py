from datetime import datetime
from dateutil.relativedelta import relativedelta

def contract_value(daily_data, injection_dates, withdrawal_dates, iw_costs, iw_rate, max_volume, monthly_storage_cost):
  '''
  input : - daily_data (dataframe): forecasted price data, with date and price colum at least (e.g. output of task1)
          - injection_dates: list of dates at which the client wishes to inject gas into a storage facility
          - withdrawal_dates: list of dates at which the client wishes to withraw gas from a storage facility
          - iw_costs (float): transaction cost fixed by contract for injection/withdrawal (in $/MMBtu)
          - iw_rate (float): monthly rate at which injections/withdrawals are performed (in millions of MMBtu per month)
          - max_volume (float): monthly maximum volume that can be injected in a storage facility (in millions of MMBtu per month)
          - monthly_storage_cost (float): monthly cost of commodity storage (in $ per month)
  output: contract value (float)
  function: compute the value of a gas storage forward contract
  '''
    contract_value = 0
    for i in range(len(injection_dates)):
        # First we retrieve from the previous code the natural gas prices from i/w dates
        i_date = injection_dates[i]
        injection_unitloss = daily_data.loc[i_date]['Prices']
        print("injection price :",injection_unitloss)
        w_date = withdrawal_dates[i]
        withdrawal_unitgain = daily_data.loc[w_date]['Prices']
        print("withdrawal price :",withdrawal_unitgain)
        # Then we count the number of storage months, assuming the storage fee only depends on the months covered
        storage_duration = relativedelta(w_date,i_date).months + 1
        print("storage duration :",storage_duration)
        # It follows that the storage cost on this period is
        storage_cost = monthly_storage_cost*storage_duration
        print("Storage cost :", storage_cost)
        # Over this period, assuming the injection/withdrawal rate is given monthly in millions of MMBtu, the stored amount of gas is
        currently_stored = min(iw_rate*storage_duration, max_volume)
        print("Stored amount for the current period :",currently_stored)
        # Which costs
        volume_saving = iw_costs*currently_stored/1000000
        # We divided by 1000000 because the price is given per million MMBtu
        print("Revenue:",currently_stored*withdrawal_unitgain)
        print("Full injection cost:", currently_stored*injection_unitloss-volume_saving)
        # Finally, we add these costs the contract value
        contract_value += currently_stored*withdrawal_unitgain - currently_stored*injection_unitloss - storage_cost + volume_saving
    return contract_value
    
# Test 

injection_dates = [datetime.strptime('2024/10/31','%Y/%m/%d')]
withdrawal_dates = [datetime.strptime('2025/02/28','%Y/%m/%d')]
iw_costs = 10000 # $ per MMBtu
iw_rate = 1000000 # million MMBtu per month
max_volume = 50000000 # million MMBtu per month
monthly_storage_cost = 100000 # $ per MMBtu per month

contract_value(injection_dates, withdrawal_dates, iw_costs, iw_rate, max_volume, monthly_storage_cost)
