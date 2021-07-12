---
layout: post
title: 1. AI trading bot--Fetching data from Binance
---

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>




This blog post is the first in the series of `building an AI (Artificial intelligence) crypto trading bot`, the chambot. 

The assumption we make here is that you are already familiar with how the stock, crypto, or foreign exchange work. If we pick crypto as our financial instrument without loss of generality, you will notice that the price of crypto is always moving up and down. This movement is risky for an investor who might incur huge losses if the crypto price drops significantly. On the other hand, traders that buy and sell crypto try to use these fluctuations in crypto prices to their advantage.

They buy when the price is low, and sell when the price is high. The real question is when do you buy (hold a position) and when do you sell (leave a position). Many traders spend hours staring at the computer screen waiting for the right moment to act.  In a couple of blog posts, I will show how to build a robot that automatically buys at the low and sell at a high price. This bot is called the chambot and uses AI for smart buying and selling. We will need data to do this,  AI models need that has predictor variables and a target. Our workflow will reflect this. 

Here is the work flow:

![](../../images/flow_diagram.jpg)

These are the steps we will take:
1. Fetch market data from Binance
2. Create a target variable
3. Create predictor variables
4. Merge the predictor variables and the target variable to form a dataset that will be used to train an AI model
5. Train AI model
6. Put the model in production on AWS to start buying and selling crypto and making huge profits. 

In this short blog post, I will show you how to fetch data from [Binance](https://www.binance.com/en) and store it as a `.csv`. 

Make sure your `requirements.txt` contains the following modules.


```python
import requests
import json 
import pandas as pd 
import numpy as np 
import datetime as dt 
from binance.client import Client
import time
from datetime import timedelta, datetime
import math
from dateutil import parser
import os.path
import config ### this config file should contain your API keys API_KEY_BINANCE and API_SECRET_BINANCE
              ### ofcourse I will not put mine here :) 
```

We will define two important functions that will fetch data and download the data. We will set the `batch_size` and the `binsizes`, as the name implies these are the batch size and the bin sizes. 


```python
binsizes = {"1m": 1, "15m": 15, "1h": 60, "1d": 1440}
batch_size = 750
pd.DataFrame(binsizes,index=[0])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1m</th>
      <th>15m</th>
      <th>1h</th>
      <th>1d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15</td>
      <td>60</td>
      <td>1440</td>
    </tr>
  </tbody>
</table>
</div>




```python
## FUNCTIONS
def minutes_of_new_data(symbol, kline_size, data, source):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    # elif source == "bitmex": old = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    # if source == "bitmex": new = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
    return old, new

def get_all_binance(symbol, kline_size, save = False):

    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)

    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, 
                                                     kline_size, 
                                                     data_df, 
                                                     source = "binance")

    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])

    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): 
        print(f'Downloading all available {kline_size} data for {symbol}. Be patient..!')
    else: 
        print(f'Downloading {delta_min} minutes of new data available for {symbol}, i.e. {available_data} instances of {kline_size} data.')
    klines =( binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S")))
    data = pd.DataFrame(klines, columns = ['timestamp', 
                                           'open', 'high', 'low', 
                                           'close', 'volume', 'close_time', 
                                           'quote_av', 'trades', 
                                           'tb_base_av', 
                                           'tb_quote_av', 'ignore' ])

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: 
        data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: 
        data_df.to_csv(f'data/{filename}')
    print('All caught up..!')
    return data_df

```

`USDT` is a stable crypto currency that mimics the US dollar in the crypto world. Often it is  tractable if you trade other volatile crypto currency like the Ethereum (`ETH`) with `USDT`. So we will first grab all `X-USDT` pairs from Binance. `X` stands for any crypto currency.


```python
tickers = requests.get('https://www.binance.com/api/v1/ticker/allPrices').text
tickers = pd.DataFrame(json.loads(tickers))['symbol'].values
tickers = tickers[[tk.find('USDT') not in [0,-1] for  tk in tickers]]
tickers
```



    array(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'BCCUSDT', 'NEOUSDT', 'LTCUSDT',
           'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'EOSUSDT', 'TUSDUSDT',
           'IOTAUSDT', 'XLMUSDT', 'ONTUSDT', 'TRXUSDT', 'ETCUSDT', 'ICXUSDT',
           'VENUSDT', 'NULSUSDT', 'VETUSDT', 'PAXUSDT', 'BCHABCUSDT',
           'BCHSVUSDT', 'USDCUSDT', 'LINKUSDT', 'WAVESUSDT', 'BTTUSDT',
           'USDSUSDT', 'ONGUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'FETUSDT',
           'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'IOSTUSDT', 'CELRUSDT',
           ], dtype=object)



We can now loop through the `tickers` list and download the data for all `XUSDT` pairs. That will take quite sometime. To download the data for a single group we do as follows:


```python
tickers=['ETHUSDT']
for symbol in tickers:
    get_all_binance(symbol, '15m', save = True)
```

Hopefully everything has worked fine. Make sure you have a `data` folder in your working directory.  In subsequent blog posts, will start building the `chambot`. The full code can be found [here](https://github.com/chambox/chambot/blob/main/binance_data_fetcher.py). 


