import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import autogen  # Assuming autogen is the framework you're using
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from openai import OpenAI
%matplotlib inline

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Placeholder imports for the conceptual framework
# !pip install pyautogen~=0.1.0
# Define your stock candidates

stocks = ['ADBE', 'NOW', 'QQQ', 'SPY', 'TMO', 'LULU', 'CRM', 'MCD',
          'PANW', 'QCOM', 'ANET', 'MSFT', 'GOOG', 'AAPL', 'AMZN', 'NVDA',
          'AMD', 'TSM', 'NFLX', 'MA', 'V', 'CRWD', 'LRCX', 'MS', 'ACN',
          'HUBS', 'UNH', 'ISRG', 'INTU', 'WMT']

startday = '2023-10-01'
latestday = '2024-02-17'
bollinger_window=20
rsi_window = 14
rsi_low = 32
rsi_upp = 85

#####################################
def fetch_stock_byday(stocks : list, start_date : str, end_date : str) -> pd.DataFrame:
    download_data = pd.DataFrame([])
    interval = '1d'
    stock_size = len(stocks)
    
    for symbol in stocks:
        stock = yf.Ticker(symbol)
        stock_data = \
        stock.history(start = start_date, end = end_date, interval= '1d').reset_index()
        if len(stock_data) > 0:
            stock_data['ticker'] = symbol
            download_data = pd.concat([download_data, stock_data])
        
    download_data = download_data.sort_values(['ticker', 'Date'], ascending = [1, 0])
    z = download_data['Date'].astype(str)
    download_data['Date'] = z
    dya_c = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    download_data = download_data[dya_c]
    tm_frame = pd.DataFrame(list(set(download_data['Date'])), columns = ['Date'])
    tm_frame = tm_frame.sort_values(['Date'], ascending = False)
    tm_frame['dayseq'] = range(1, len(tm_frame) + 1)
    download_data = pd.merge(download_data, tm_frame, on= ['Date'] , how='inner')
    download_data['Date'] = download_data['Date'].str.slice(0,10)
    download_data = download_data.sort_values(['ticker', 'Date'], ascending = [False, False])
    return download_data

def calculate_rsi(series, window = rsi_window):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
    
def calculate_bollinger_bands(series, window = bollinger_window):
        middle_band = series.rolling(window=window).mean()
        std_dev = series.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        return middle_band, upper_band, lower_band

def ind_df()  -> float:
    
    df = fetch_stock_byday(stocks, startday, latestday)
    df.sort_values(by='Date', ascending=True, inplace=True)  # Ensure data is sorted by date
    
    tickers = df['ticker'].unique()
    indicator_df_list = []
    
    for ticker in tickers:
        df_ticker = df[df['ticker'] == ticker].sort_values(by='Date', ascending=True)
        df_ticker['RSI'] = calculate_rsi(df_ticker['Close'], 14)
        df_ticker['Middle_Band'], df_ticker['Upper_Band'], df_ticker['Lower_Band'] = calculate_bollinger_bands(df_ticker['Close'], 20)
        recent_df = df_ticker[-15:]
        indicator_df_list.append(recent_df)
    
    final_df = pd.concat(indicator_df_list)
    return final_df

def plot_stock_with_indicators(DF, symbol):
    
    df = DF[DF['ticker'] == symbol]    
    df = df.sort_values(['Date'])
    
    # Setting up the figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Stock Price, Bollinger Bands, and RSI')

    # Plotting the stock price and Bollinger Bands on the first subplot
    axes[0].plot(df['Date'], df['Close'], label='Close', color='blue', lw=2)
    axes[0].plot(df['Date'], df['Middle_Band'], label='Middle Band', color='gray', linestyle='--')
    axes[0].plot(df['Date'], df['Upper_Band'], label='Upper Band', color='red', linestyle='--')
    axes[0].plot(df['Date'], df['Lower_Band'], label='Lower Band', color='green', linestyle='--')
    axes[0].fill_between(df['Date'], df['Upper_Band'], df['Lower_Band'], color='grey', alpha=0.1)
    axes[0].set_ylabel('Close')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)

    # Plotting RSI on the second subplot
    axes[1].plot(df['Date'], df['RSI'], label='RSI', color='purple', lw=2)
    axes[1].axhline(70, linestyle='--', color='red', lw=1)
    axes[1].axhline(30, linestyle='--', color='green', lw=1)
    axes[1].fill_between(df['Date'], 70, 30, color='grey', alpha=0.1)
    axes[1].set_ylabel('RSI')
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    # Setting the x-axis label
    plt.xlabel('Date')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def make_trading_decision() -> pd.DataFrame:
    indicator_df = ind_df()
    recent_day = indicator_df.Date.max()
    cols = ['ticker','Date', 'RSI', 'Middle_Band', 'Upper_Band', 'Lower_Band']
    rsi_df = indicator_df[indicator_df.Date == recent_day][cols]
    rsi_df_select = rsi_df[(rsi_df.RSI<32) | (rsi_df.RSI>85)]
    
    if len(rsi_df_select) > 0:
        rsi_df_select['suggestion'] = rsi_df_select['RSI'].apply(lambda x: 'buy' if x < rsi_low else ('sell' if x > rsi_upp else 'hold'))
        return rsi_df_select
    else:
        return pd.DataFrame([])
    
def is_termination_msg(data):
    has_content = "content" in data and data["content"] is not None
    return has_content and "TERMINATE" in data["content"]

###################################
# Define your function map
function_map = {
      "fetch_stock_byday": fetch_stock_byday,
      "calculate_rsi" : calculate_rsi,
      "calculate_bollinger_bands" : calculate_bollinger_bands,
      "plot_stock_with_indicators" : plot_stock_with_indicators,
      "ind_df": ind_df,
      "make_trading_decision": make_trading_decision
}

# Configuration for the AutoGen environment (Placeholder for actual config)
config_list =[{ "model": "gpt4",
               "api_type": "azure",
               "base_url": "https://******-gpt4.openai.azure.com/", 
               "api_key":"###########",
               'api_version':'2023-11-01-preview'}]

# Define the system prompt (Placeholder text)
SYSTEM_PROMPT = """You are an intelligent assistant capable of analyzing stock data. 
Use the provided functions to fetch data, calculate RSI, make trading decisions, 
and evaluate stocks based on predefined criteria. Please conclude with 
"TERMINATE" once you have successfully answered the user's request.
"""

llm_config = {
    "config_list": config_list,
    "temperature": 0.3,
    "functions": [
        {
            "name": "make_trading_decision",
            "description": "provide trading suggestion based on RSI indicator",
            "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            },
        },
    ]
}
    

work_dir = "C:\\stock\\autogen"
config_list_user = {
    "work_dir": ,
    "use_docker": False
}

if not os.path.exists(work_dir):
    os.makedirs(work_dir)
os.chdir(work_dir)    

# Initialize the UserProxyAgent
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode='NEVER', 
    code_execution_config = config_list_user,
    max_consecutive_auto_reply=10
)

# register the function names with the UserProxyAgent
user_proxy.register_function(function_map = function_map)

# Initialize the AssistantAgent with the system message and configuration
stock_analyst = AssistantAgent(
    name="stock_analyst",
    system_message = SYSTEM_PROMPT,
    llm_config = llm_config,
    is_termination_msg=is_termination_msg,
    code_execution_config=False
)

################################
message1 = """Evaluate stock recommendations, also list the bollinger_bands of all selected stock"""

user_proxy.initiate_chat(stock_analyst, message = message1 )   

################################
message2 = """after Evaluate stock recommendations, provide python code for creating a chart of  bollinger_bands and RSI  
 of top selected stock, i.e. the lowest RSI stock, save the chart into my work dir """

user_proxy.initiate_chat(stock_analyst, message = message2 ) 


