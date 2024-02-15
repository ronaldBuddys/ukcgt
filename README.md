# UK Capital Gains Tax Calculation  

**Disclaimer: This package is for informational purposes only and is not 
 as a substitute for professional accounting advice or services. The author is not liable for its accuracy 
or use. Users should consult professional advisors for tax-related decisions. 
Use at your own risk.**

## Quick Start



for an expanded version of below see: [capitalgains/examples/capital_gains_run.py](capitalgains/examples/capital_gains_run.py)
```python

import pandas as pd

from capitalgains import get_data_path
from capitalgains.capital_gains import CapitalGains

# CapitalGains object contains methods for calculation
cg = CapitalGains()

# read in fx rates
fx = pd.read_csv(get_data_path("example_fx_rates.csv"))
fx['date'] = fx['date'].values.astype('datetime64[D]')

# read in trades
td = pd.read_csv(get_data_path("example_trades.csv"), dtype={'tradeid': str})
td['date'] = td['date'].values.astype('datetime64[D]')

# run pnl calculations on trades using fx and applying rules
net, split = cg.run(td, fx, pnl_ccy="GBP")

print(pd.pivot_table(net,
                     index='tax_year',
                     values='pnl',
                     aggfunc='sum'))

```

# Data

## Trades

Trades are expected to be in the following format

| buy_ccy   | sell_ccy   |   buy_amount |   sell_amount | date       |   tradeid |
|:----------|:-----------|-------------:|--------------:|:-----------|----------:|
| CAD       | LTC        |    447.689   |      9.90062  | 2018-12-28 |     00188 |
| USD       | CAD        |   6549.77    |   8396.01     | 2022-04-27 |     00546 |
| XRP       | CAD        |   2841.55    |   1135.37     | 2020-08-14 |     00377 |
| XRP       | ETH        |    676.594   |      0.650937 | 2020-05-22 |     00360 |
| ETH       | MIOTA      |      1.89045 |   5412.35     | 2021-10-05 |     00502 |

when read in should have dtypes:

| Column Name | Data Type       |
|:------------|:----------------|
| buy_ccy     | object (string) |
| sell_ccy    | object (string) |
| buy_amount  | float64         |
| sell_amount | float64         |
| date        | datetime64[s]   |
| tradeid     | object (string) |



 
The below is a code snippet using the values from `data/example_trades.csv`

```python

import pandas as pd
from capitalgains import get_data_path

td = pd.read_csv(get_data_path("example_trades.csv"), dtype={'tradeid': str})
td['date'] = td['date'].values.astype('datetime64[D]')
```


## FX Rates

This is required to convert any transaction (trade) that does not have the
PnL Currency (GBP) in either leg. 


| base   | quote   | date                |   rate |
|:-------|:--------|:--------------------|-------:|
| BTC    | GBP     | 2011-09-06 00:00:00 |  4.254 |
| BTC    | GBP     | 2011-09-07 00:00:00 |  4.261 |
| BTC    | GBP     | 2011-09-08 00:00:00 |  4.206 |
| BTC    | GBP     | 2011-09-09 00:00:00 |  3.161 |
| BTC    | GBP     | 2011-09-10 00:00:00 |  3     |


The dtypes of the above table (DataFrame) should be:

| Column Name | Data Type       |
|-------------|-----------------|
| base        | object (string) |
| quote       | object (string) |
| date        | datetime64[s]   |
| rate        | float64         |
 
The below is a code snippet using the values from `data/example_fx_rates.csv`

```python

import pandas as pd
from capitalgains import get_data_path

fx = pd.read_csv(get_data_path("example_fx_rates.csv"))
fx['date'] = fx['date'].values.astype('datetime64[D]')

```

## Capital Gains Tax Quick Overview 

`CaptialGains.run(...)` does the following:

1) split trades - any trades with GBP in either leg is split, using fx rates
2) aggregate daily total amounts bought and sold per asset
3) apply rules: Same Day, Bed and Breakfast, Section 104 (average pooling)

**NOTE: for trades that need to be split it's assumed the fx rates are triangulated,
meaning it shouldn't matter if the buy or sell currency is used to get the GBP
equivalent for each leg. This might not be the case as, fx rates only have a single value per day, which could have been snapped at 
a time different to when a trade was made. You may want to split trades yourself prior to providing to run(...).**



## virtual setup environment

recommended to use `python >= 3.10` and Linux or macOS.

from directory containing `requirements.txt`: create virtual environment

`python -m venv venv`

activate virtual environment

`source venv/bin/activate`

install packages

`pip install -r requirements.txt`

(optional) install package in editable mode

`pip install -e .`


