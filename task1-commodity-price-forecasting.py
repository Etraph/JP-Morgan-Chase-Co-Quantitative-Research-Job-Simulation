import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.interpolate import PchipInterpolator

#----------------------------------------------------
# Data extraction & visualisation of seasonal trends
#----------------------------------------------------
'''
The "Nat_Gas.csv" file must contain daily dates and prices for a commodity, which in this case has seasonal trends.
'''
ng_data = pd.read_csv('Nat_Gas.csv')
dts = ng_data['Dates']
prc = ng_data['Prices']
var = np.max(prc)-np.min(prc)
plt.figure(figsize = (14,8))
plt.scatter(dts,prc,color = 'purple',label = 'monthly commodity price from 10/2020 to 09/2024')

for i in range(len(dts)-1):
    """
    To better visualize the correspondance to seasonal trends, this function colors the up- and down-trends, faced with dates
    gapped by 3 months to showcase each season
    """
    start, end = dts[i], dts[i + 1]
    prc_start, prc_end = prc[i], prc[i + 1]
    color = "red" if prc_end > prc_start else "blue"
    # Fill between curve and x-axis
    plt.fill_between([start, end], [prc_start, prc_end], color=color, alpha=0.4)   
plt.xticks(dts[2::3])

#----------------------------------------
# Observation of local minima and maxima
#----------------------------------------

plt.vlines(x=dts[8::12], ymin=np.min(prc)-var/3, ymax=np.max(prc)+var/3, colors='green', linestyles='dashed', label="mid-year")
plt.vlines(x=dts[2::12], ymin=np.min(prc)-var/3, ymax=np.max(prc)+var/3, colors='orange', linestyles='dashed', label="year")
plt.ylim(np.min(prc)-var/3,np.max(prc)+var/3)
plt.legend()
plt.show()

#-------------------
# Price forecasting 
#-------------------
'''
Example : for natural gas storage, one expects a seasonal trend. Suppose it overall increases.
It is set yearly to capture a full cycle and simulated using a Holt-Winters exponential smoothing model, considering an upward-trending seasonality.
If one has an increasing trend, one can increase precision by adding a trend parameter, which can be removed in the following:
'''

seasonal_parameter = 12        # Number of months characterising seasonality in given data
model = ExponentialSmoothing(ng_data['Prices'], trend='add', seasonal='add', seasonal_periods=seasonal_parameter)        # With upward trend
#model = ExponentialSmoothing(ng_data['Prices'], seasonal='add', seasonal_periods=seasonal_parameter)        # Without
model_fit = model.fit()

forecast_start_date = '10/31/24'        # Last day included in the given data
num_periods = 16        # number of periods considered in the data
dt_freq = "M"         # data frequency (here monthly)
# Then forecast the next year with the fit model
future_dates = pd.date_range(start=forecast_start_date,periods = num_periods, freq=dt_freq)
forecast = model_fit.forecast(steps=16)

# Now set in place a dataframe for the forecasted monthly data
forecast_data = pd.DataFrame({"Dates": future_dates, "Prices": forecast})
monthly_data = pd.concat([ng_data,forecast_data])

# Finally, generate a daily date range to capture the entire period
start_date, end_date = monthly_data["Dates"].min(), monthly_data["Dates"].max()
daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")

# And interpolate to obtain the corresponding daily prices
interp_func = PchipInterpolator(monthly_data["Dates"], monthly_data["Prices"])
daily_prices = interp_func(daily_dates)

# TNow conclude with a daily DataFrame
daily_data = pd.DataFrame({"Dates": daily_dates, "Prices": daily_prices})

# Give the user an input
input_date = input("Enter a date: ")
input_index = daily_data[daily_data['Dates'] == input_date].index
# Return the desired price
print("Predicted price : ",daily_data.loc[input_index])

# Also plot the data for better visualisation
future = ng_data["Dates"].max()
mask = daily_data["Dates"] <= future        # to differentiate interpolated past data and forecasted data
tpast, ppast = daily_data["Dates"][mask], daily_data["Prices"][mask]        # Past data
tforecast, pforecast = daily_data["Dates"][~mask], daily_data["Prices"][~mask]        # Forecast data
plt.figure(figsize = (14,8))
plt.plot(tpast, ppast, label="Interpolated Daily Prices")
plt.plot(tforecast, pforecast, label="Forecast Daily Prices")
plt.scatter(dts,prc,color = 'purple',label = 'raw data')
plt.title("daily estimated natural gas prices with a 2025 forecast")
plt.legend()
plt.legend()
plt.show()

