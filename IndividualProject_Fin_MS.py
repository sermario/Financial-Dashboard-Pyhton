# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:19:10 2021 

@author: mserrano
"""


# Individual Project - Financial Dashboard

#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import streamlit as st
import matplotlib.dates as mpl_dates
import matplotlib.ticker as ticker
import matplotlib as mpl
import mplfinance as fplt


st.set_page_config(layout="wide")

# main data collected from https://finance.yahoo.com/
# yahoo_fin functions from http://theautomatic.net/yahoo_fin-documentation/#get_company_info 
# streamlit functions from https://docs.streamlit.io/

#==============================================================================

def tab1():
    
    # Add dashboard title and description
    st.title("Financial Dashboard")
    st.write("Author: Mario Serrano")
    st.write("Data source: Yahoo Finance")
    st.header(company_name+' Summary')
    
    col1, col2 = st.columns([1,6])
    
    #radio buttons time
    #time period
    period = col1.radio('Period', ('1M','3M','6M','YTD','1Y','5Y','Max'))
    
    def period_start_date(per):
        if period =='1M':
            return datetime.now().date() - timedelta(days=30)
        elif period =='3M':
            return datetime.now().date() - timedelta(days=30*3)
        elif period =='6M':
            return datetime.now().date() - timedelta(days=30*6)
        elif period =='YTD':
            return datetime.now().date() - timedelta((datetime.today()-datetime(datetime.today().year,1,1)).days)
        elif period =='1Y':
            return datetime.now().date() - timedelta(days=30*12)
        elif period =='5Y':
            return datetime.now().date() - timedelta(days=30*12*5)
        elif period =='Max':
            return datetime.now().date() - timedelta(days=30*12*50)
    
    #get data for plot 
    @st.cache
    def GetData(ticker,start_date):
        return si.get_data(ticker,start_date=start_date, end_date=datetime.now().date())

    # create plot with close price and volume
    if ticker != '-':
        stock_price = GetData(ticker, start_date=period_start_date(per=period))
        
        fig, ax1 = plt.subplots(figsize=(15, 7.5))
        
        #plot1 stock price
        ax1.plot(stock_price.index,stock_price['adjclose'],color="navy")
        ax1.set_ylim(stock_price['adjclose'].min()*0.99,stock_price['adjclose'].max()*1.01)
        ax1.fill_between(stock_price.index,stock_price['adjclose'],color="royalblue")
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax1.set_title('Adjusted close price and volume for '+company_name,fontsize=20)
        ax1.set_ylabel('Ajusted close price (USD)',fontsize=17)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.grid(linestyle ='--',linewidth=1,color='gainsboro')
        plt.xticks(rotation=45)


        #inside plot2 volume
        ax2 = ax1.twinx()
        ax2.bar(stock_price.index,stock_price['volume']/1000000,color="lightsteelblue")
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.set_ylim(0,stock_price['volume'].max()*6/1000000)
        ax2.set_ylabel('Volume (millions)',fontsize=17)
        ax2.spines['top'].set_visible(False) #https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        col2.pyplot(fig)
        
    # Add table to show quote table
    @st.cache
    def GetQuoteTable(ticker):
        return pd.DataFrame(si.get_quote_table(ticker),index=[0]).transpose()
    
    #output data
    if ticker != '-':
        info = GetQuoteTable(ticker)
        info.iloc[:,0] = info.iloc[:,0].astype(str)
        info.columns = ['Value']
        col2.dataframe(info.style.set_properties(**{'text-align': 'right'}),height=1000,width=500) #https://stackoverflow.com/questions/17232013/how-to-set-the-pandas-dataframe-data-left-right-alignment
        
    
# Tab 2
#==============================================================================

def tab2():
    
    # Add dashboard title and description
    st.title("Financial Dashboard")
    st.write("Author: Mario Serrano")
    st.write("Data source: Yahoo Finance")
    st.header(company_name+' Chart')
    
    col1, col2, col3, col4 = st.columns([0.5,1,1,1])
    
    #dates and periods
    period = col1.radio('Period', ('1M','3M','6M','YTD','1Y','5Y','Max'))
    def period_start_date(per):
        if period =='1M':
            return 30
        elif period =='3M':
            return 30*3
        elif period =='6M':
            return 30*6
        elif period =='YTD':
            return (datetime.today()-datetime(datetime.today().year,1,1)).days
        elif period =='1Y':
            return 365
        elif period =='5Y':
            return 365*5

    if ticker != '-':
        if period == "Max":
            start_date_t2 = col2.date_input("Start",si.get_data(ticker).index.min())
        elif period != "Max":
            start_date_t2 = col2.date_input("Start", datetime.today().date() - timedelta(days=period_start_date(period)))
    
    if period == "Max":
        end_date_t2 = col2.date_input("End", datetime.today().date())
    elif period != "Max":
        end_date_t2 = col2.date_input("End", start_date_t2 + timedelta(days=period_start_date(period)))
    
   
    #time interval
    timeInterval = col3.radio('Time Interval', ('Day','Month','Year'))
    
    if timeInterval == "Day":
        selectedDates = pd.date_range(start_date_t2, end_date_t2, freq="D")
    if timeInterval == "Month":
        if period == "1M":
            selectedDates = pd.date_range(start_date_t2, end_date_t2, freq="D")
        elif period != "1M":    
            selectedDates = pd.date_range(start_date_t2, end_date_t2, freq="M")
    if timeInterval == "Year":
        if len(pd.date_range(start_date_t2, end_date_t2, freq="Y"))<1:
            selectedDates = pd.date_range(start_date_t2, end_date_t2, freq="D")
        elif len(pd.date_range(start_date_t2, end_date_t2, freq="Y"))>=1:    
            selectedDates = pd.date_range(start_date_t2, end_date_t2, freq="Y")
        
    
    #get data for plot
    @st.cache
    def GetData(ticker,start_date,end_date):
        return si.get_data(ticker,start_date=start_date, end_date=end_date)

    #plot type    
    typePlot = col4.radio('Plot Type', ('Lines','Candle'))

    if typePlot =='Lines':    
        st.write("Lines Plot for "+company_name,fontsize=20)
    elif typePlot =='Candle': 
        st.write("Candle Plot for "+company_name,fontsize=20)
    
    #date formatter
    if period == "1M" or period == '3M' or period == '6M':
        myFmt = mpl_dates.DateFormatter("%d-%b-%Y")
    else: 
        myFmt = mpl_dates.DateFormatter("%b-%Y")
    
    #width of bars
    if timeInterval =='Day':
        w_bars = 1
    elif timeInterval =='Month' and (period == 'YTD' or period == '1Y'or period == '5Y'or period == 'Max') :
        w_bars = 30
    elif timeInterval =='Year' and (period == 'YTD'or period == '1Y'or period == '5Y'or period == 'Max'):
        w_bars = 360
    else:
        w_bars = 1
    
    # create plot with close price and volume
    
    if ticker != '-':

        stock_price = GetData(ticker,start_date=start_date_t2,end_date=end_date_t2)
        stock_price = stock_price[stock_price.index.isin(selectedDates)]
         
        if typePlot =='Lines':
    
            fig, ax1 = plt.subplots(figsize=(15, 8))
            
            #plot1 stock price
            ax1.plot(stock_price.index,stock_price['adjclose'],color="navy",label='Close Price')
            ax1.plot(stock_price.index,stock_price['adjclose'].rolling(window =50).mean(),
                     color="royalblue",label='SMA 50 days')
            ax1.set_ylim(stock_price['adjclose'].min()*0.99,stock_price['adjclose'].max()*1.01)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            plt.xticks(rotation=45)
            ax1.xaxis.set_major_formatter(myFmt)
            ax1.set_ylabel('Ajusted close price (USD)',fontsize=17)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.grid(linestyle ='--',linewidth=1,color='gainsboro')
            ax1.legend(loc='upper left')
            
            #inside plot2 volume
            ax2 = ax1.twinx()
            ax2.bar(stock_price.index,stock_price['volume']/1000000,
                    color="lightsteelblue",width=w_bars)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            ax2.set_ylim(0,stock_price['volume'].max()*6/1000000)
            ax2.set_ylabel('Volume (millions)',fontsize=17)
            ax2.xaxis.set_major_formatter(myFmt)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            
            st.pyplot(fig)
        
        elif typePlot =='Candle':
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(fplt.plot(stock_price,
                                type='candle',
                                ylabel='Ajusted close price (USD)',
                                volume=True,
                                ylabel_lower='Volume (millions)',
                                style ='yahoo',
                                figsize=(15, 7.5)))
        
# Tab 3
#==============================================================================

def tab3():
    
    # Add dashboard title and description
    st.title("Financial Dashboard")
    st.write("Author: Mario Serrano")
    st.write("Data source: Yahoo Finance")
    st.header(company_name+' Statistics')
    
    
    if ticker != '-':
        info = si.get_stats(ticker)
        
        #Valuation Measures
        info_vm = si.get_stats_valuation(ticker)
        info_vm.columns = ['','Value']
        
        #Stock Price History
        
        info_sp = info.iloc[0:7,:].set_index('Attribute') #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html
        
        #Fiscal Year
        info_fy = info.iloc[29:31,:].set_index('Attribute')
        
        #Share Statistics
        info_st = info.iloc[7:19,:].set_index('Attribute')
        
        #Profitability
        info_pr = info.iloc[31:33,:].set_index('Attribute')
        
        #Management Effectiveness
        info_me = info.iloc[33:35,:].set_index('Attribute')
        
        #Income Statement
        info_is = info.iloc[35:43,:].set_index('Attribute')
        
        #Balance Sheet
        info_bs = info.iloc[43:48,:].set_index('Attribute')
        
        #Chashflow Statement
        info_cs = info.iloc[48:,:].set_index('Attribute')
        
        #Dividends and splits
        info_ds = info.iloc[19:29,:].set_index('Attribute')
        
        #split columns
        col1, col2, = st.columns([0.9,1.1])
        
        col1.subheader("Valuation Measures")
        col1.dataframe(info_vm.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col1.subheader("Financial Highlights")

        col1.write("Fiscal Year")
        col1.dataframe(info_fy.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col1.write("Profitability")
        col1.dataframe(info_pr.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col1.write("Management Effectiveness")
        col1.dataframe(info_me.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col1.write("Income Statement")
        col1.dataframe(info_is.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col1.write("Balance Sheet")
        col1.dataframe(info_bs.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col1.subheader("Chashflow Statement")
        col1.dataframe(info_cs.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col2.subheader("Trading Information")
        
        col2.write("Stock Price History")
        col2.dataframe(info_sp.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col2.write("Share Statistics")
        col2.dataframe(info_st.style.set_properties(**{'text-align': 'right'}),height=1000)
        
        col2.write("Dividends & Splits")
        col2.dataframe(info_ds.style.set_properties(**{'text-align': 'right'}),height=1000)
         

# Tab 4
#==============================================================================

def tab4():
    
    # Add dashboard title and description
    st.title("Financial Dashboard")
    st.write("Author: Mario Serrano")
    st.write("Data source: Yahoo Finance")
    st.header(company_name+' Financials')
    
    col1, col2, col3 =  st.columns([1,1,1])

    #buttons
    income = col1.button('Income Statement')
    balance = col2.button('Balance Sheet')
    cash = col3.button('Cash Flow')
    
    if ticker != '-':
        if income:
            info1=pd.DataFrame.from_dict(si.get_financials(ticker, yearly = False, quarterly = True)['quarterly_income_statement'])
            info1.columns = pd.to_datetime(info1.columns, format = '%m/%d/%Y').strftime('%Y-%m-%d')
            st.dataframe(info1,height=1000)
        elif balance:
            info1=pd.DataFrame.from_dict(si.get_financials(ticker, yearly = False, quarterly = True)['quarterly_balance_sheet'])
            info1.columns = pd.to_datetime(info1.columns, format = '%m/%d/%Y').strftime('%Y-%m-%d')
            st.dataframe(info1,height=1000)
        elif cash:
            info1=pd.DataFrame.from_dict(si.get_financials(ticker, yearly = False, quarterly = True)['quarterly_cash_flow'])
            info1.columns = pd.to_datetime(info1.columns, format = '%m/%d/%Y').strftime('%Y-%m-%d')
            st.dataframe(info1,height=1000)
            
            
# Tab 5
#==============================================================================

def tab5():
    
    # Add dashboard title and description
    st.title("Financial Dashboard")
    st.write("Author: Mario Serrano")
    st.write("Data source: Yahoo Finance")
    st.header(company_name+' Analysis')
    
    if ticker != '-':
        info = si.get_analysts_info(ticker)
        
        #Earnings Estimate
        st.write('Earnings Estimate')
        info_ee = pd.DataFrame.from_dict(info['Earnings Estimate'])
        info_ee = info_ee.set_index('Earnings Estimate')
        st.dataframe(info_ee,height=1000,width = 1000)
        
        #Revenue Estimate
        st.write('Revenue Estimate')
        info_re = pd.DataFrame.from_dict(info['Revenue Estimate'])
        info_re = info_re.set_index('Revenue Estimate')
        st.dataframe(info_re,height=1000,width = 1000)
        
        #Earnings History
        st.write('Earnings History')
        info_eh = pd.DataFrame.from_dict(info['Earnings History'])
        info_eh = info_eh.set_index('Earnings History')
        st.dataframe(info_eh,height=1000,width = 1000)
        
        #EPS Trend
        st.write('EPS Trend')
        info_et = pd.DataFrame.from_dict(info['EPS Trend'])
        info_et = info_et.set_index('EPS Trend')
        st.dataframe(info_et,height=1000,width = 1000)
        
        #EPS Revisions
        st.write('EPS Revisions')
        info_epr = pd.DataFrame.from_dict(info['EPS Revisions'])
        info_epr = info_epr.set_index('EPS Revisions')
        st.dataframe(info_epr,height=1000,width = 1000)
        
        #Growth Estimates
        st.write('Growth Estimates')
        info_ge = pd.DataFrame.from_dict(info['Growth Estimates'])
        info_ge = info_ge.set_index('Growth Estimates')
        st.dataframe(info_ge,height=1000,width = 1000)
        

# Tab 6
#==============================================================================

def tab6():
    
    # Add dashboard title and description
    st.title("Financial Dashboard")
    st.write("Author: Mario Serrano")
    st.write("Data source: Yahoo Finance")
    st.header(company_name+' Monte Carlo Simulation')
    
    col1, col2, col3, col4, col5, col6 =  st.columns([0.5,0.75,0.75,0.5,1,1])
    
    #dates and periods
    period = col1.radio('Period', ('6M','1Y','2Y','5Y'))
    def period_start_date(per):
        if period =='6M':
            return 30*6
        elif period =='1Y':
            return 365
        elif period =='2Y':
            return 365*2
        elif period =='5Y':
            return 365*5

    start_date_t2 = col2.date_input("Start Date", datetime.today().date() - timedelta(days=period_start_date(period)))

    end_date_t2 = col2.date_input("End Date", start_date_t2 + timedelta(days=period_start_date(period)))
    
    #simulations
    nsim = col3.selectbox("Number of Simulations",(200,500,1000))
    timehor = col3.selectbox("Time Horizon (Days)",(30,60,90))
    
    @st.cache(allow_output_mutation=True)
    class MonteCarlo(object): ### from Financial Programming IESEG Section 3 Jupyter notebook
    
        def __init__(self, ticker, start_date, end_date, time_horizon, n_simulation, seed):
            
            # Initiate class variables
            self.ticker = ticker  # Stock ticker
            self.start_date = start_date # Text, YYYY-MM-DD
            self.end_date = end_date  # Text, YYYY-MM-DD
            self.time_horizon = time_horizon  # Days
            self.n_simulation = n_simulation  # Number of simulations
            self.seed = seed  # Random seed
            self.simulation_df = pd.DataFrame()  # Table of results
            
            # Extract stock data
           
            self.stock_price = si.get_data(ticker, self.start_date, self.end_date)
            
            # Calculate financial metrics
            # Daily return (of close price)
            self.daily_return = self.stock_price['close'].pct_change()
            # Volatility (of close price)
            self.daily_volatility = np.std(self.daily_return)
            
        def run_simulation(self):
            
            # Run the simulation
            np.random.seed(self.seed)
            self.simulation_df = pd.DataFrame()  # Reset
            
            for i in range(self.n_simulation):
    
                # The list to store the next stock price
                next_price = []
    
                # Create the next stock price
                last_price = self.stock_price['close'][-1]
    
                for j in range(self.time_horizon):
                    
                    # Generate the random percentage change around the mean (0) and std (daily_volatility)
                    future_return = np.random.normal(0, self.daily_volatility)
    
                    # Generate the random future price
                    future_price = last_price * (1 + future_return)
    
                    # Save the price and go next
                    next_price.append(future_price)
                    last_price = future_price
    
                # Store the result of the simulation
                self.simulation_df[i] = next_price
    
        def plot_simulation_price(self):
            
            # Plot the simulation stock price in the future
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 7.5, forward=True)
    
            plt.plot(self.simulation_df)
            plt.title('Monte Carlo simulation for ' + company_name +' stock price in next ' + str(self.time_horizon) + ' days')
            plt.xlabel('Days',fontsize=15)
            plt.ylabel('Price',fontsize=15)
    
            plt.axhline(y=self.stock_price['close'][-1], color='red')
            plt.legend(['Stock price at ' + str(end_date_t2) + ' is: ' + str(np.round(self.stock_price['close'][-1], 2))])
            ax.get_legend().legendHandles[0].set_color('red')
    
            return plt.show()
        
        def plot_simulation_hist(self):
            
            # Get the ending price of the 200th day
            ending_price = self.simulation_df.iloc[-1:, :].values[0, ]
    
            # Plot using histogram
            
            plt.hist(ending_price, bins=50)
            plt.xlabel('Future Prices (USD)',fontsize=15)
            plt.ylabel('Frequency',fontsize=15)
            plt.axvline(x=self.stock_price['close'][-1], color='red')
            plt.legend(['Stock price at '+ str(end_date_t2) +' is: ' + str(np.round(self.stock_price['close'][-1], 2))])
            
            
            return plt.show()
        
        def value_at_risk(self):
            # Price at 95% confidence interval
            future_price_95ci = np.percentile(self.simulation_df.iloc[-1:, :].values[0, ], 5)
    
            # Value at Risk
            VaR = self.stock_price['close'][-1] - future_price_95ci
            return np.round(VaR, 2)
    
    if ticker != '-':
        
        mc_sim = MonteCarlo(ticker=ticker,start_date=start_date_t2, end_date=end_date_t2,
                    time_horizon=timehor, n_simulation=nsim, seed=100)
        mc_sim.run_simulation()
        col5.metric('Price at ' + str(end_date_t2) + ' (USD)','$'+str(np.round(mc_sim.stock_price['close'][-1],2)))
        col5.metric('VaR 95% Confidence','$'+str(mc_sim.value_at_risk()))
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader('Simulations Plot')
        st.pyplot(mc_sim.plot_simulation_price())
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader('Simulations Histogram')
        st.pyplot(mc_sim.plot_simulation_hist())
        
        
        
# Main body
#==============================================================================

def run():
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = ['-'] + si.tickers_sp500()
    
    # Add selection box
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    
    global company_name
    if ticker != '-':
        company_name = si.get_quote_data(ticker)['shortName']
    elif ticker == '-': 
        company_name = 'None Selected'
        
     
    #update info button
    st.sidebar.button("Update Info")
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Summary', 'Chart','Statistics','Financials','Analysis','Monte Carlo Simulation'])
    
    # Show the selected tab
    if select_tab == 'Summary':
        # Run tab 1
        tab1()
    elif select_tab == 'Chart':
        # Run tab 2
        tab2()
    elif select_tab == 'Statistics':
        # Run tab 3
        tab3()
    elif select_tab == 'Financials':
        # Run tab 4
        tab4()
    elif select_tab == 'Analysis':
        # Run tab 5
        tab5()
    elif select_tab == 'Monte Carlo Simulation':
        # Run tab 6
        tab6()

    
if __name__ == "__main__":
    run()
    
# END
###############################################################################