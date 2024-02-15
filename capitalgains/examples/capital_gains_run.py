
import os
import pandas as pd

from capitalgains import get_data_path
from capitalgains.capital_gains import CapitalGains
from capitalgains import get_path

pd.set_option('display.max_columns', 100)

# ---
# parameters
# ---

pnl_ccy = 'GBP'
# tax year (ending in) - in example data 2022 has more trade ccy/assest
tax_year = 2022

# --
# Initialise a CapitalGains object
# --

cg = CapitalGains()

# --
# get fx rates
# --

fx = pd.read_csv(get_data_path("example_fx_rates.csv"))
fx['date'] = fx['date'].values.astype('datetime64[D]')
# example data can have rates = 0
fx = fx.loc[fx['rate'] > 0]

# --
# get trades
# --

td = pd.read_csv(get_data_path("example_trades.csv"), dtype={'tradeid': str})
td['date'] = td['date'].values.astype('datetime64[D]')

# ---
# run pnl calculations on trades using fx and applying rules
# ---

# get the trades netted by date, buy_ccy or sell_ccy, buy/sell
# - and how to trades were split - contains mapping from tradeid to agg_id (aggregation index)
net, split = cg.run(td, fx, pnl_ccy=pnl_ccy)

print(pd.pivot_table(net,
                     index='tax_year',
                     values='pnl',
                     aggfunc='sum'))

# ------------
# (optional) write results for specific tax year to file
# ------------

# below is a bit of a mess, left over from old scripts...

# directory and file names to write results to
output_base_dir = get_path("documents")
os.makedirs(output_base_dir, exist_ok=True)

out_dir = os.path.join(output_base_dir, f"{tax_year}_tax_year_results")
out_file_base = f"{tax_year - 1}-{tax_year} UK capital gains calculation"

# ---
# get summary for take year (needed for completing self-assessment)
# ---

print("-" * 50)
print(f"summing total sales for tax_year: {tax_year}")
# take desired tax year only
ty = net.loc[net['tax_year'] == tax_year]

assert len(ty) > 0, f"there are no trades in this tax year ({tax_year})"

# get the summary break-downs
summary, det_summary, det_sum_breakdown = cg.get_summary_for_ty_ccy(ty, pnl_ccy)

print("-" * 25)
print("summary:")
print(summary)
print("-" * 25)
print("detailed summary:")
print(det_summary)

# ---
# generate tex report (and write DataFrames to file)
# ---

print("-" * 50)
cg.write_results_to_tex(net, summary, det_summary, det_sum_breakdown, out_dir, out_file_base)

data_to_write = {
    "split_trades": split,
    "input_trades": td,
    "fx_rates": fx
}

for k, v in data_to_write.items():
    out_file = os.path.join(out_dir, f"{k}.csv")
    print(f"writing {k} to: {out_file}")
    v.to_csv(out_file, index=False)


