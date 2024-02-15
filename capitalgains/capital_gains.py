"""
CapitalGains class - used for calculating losses, gains and corresponding tax obligations for UK Tax
"""

import os
import re
import pandas as pd
import numpy as np

import warnings

from capitalgains import get_data_path

from pylatex.utils import NoEscape
from pylatex import Document, Package, Command
from capitalgains.utils import dataframe_to_tex

class CapitalGains:
    """capital gains class"""

    _data_dir = get_data_path()

    def __init__(self):
        pass

    @staticmethod
    def drop_trades_missing_ccy(df):
        """Drop trades missing values in buy/sell_ccy"""
        drop_trades = np.zeros(len(df), dtype=bool)

        # missing could be null, or ""
        for _ in ['buy', 'sell']:
            drop_trades += pd.isnull(df[f'{_}_ccy'])
            drop_trades += df[f'{_}_ccy'] == ""

        return df.loc[~drop_trades], df.loc[drop_trades]

    @staticmethod
    def split_trades(td, fx, pnl_ccy='GBP', include_rate=False, verbose=True):
        """
        Splits and converts a DataFrame of trades (td) into the target profit and loss currency (pnl_ccy),
        using FX rates from another DataFrame (fx). The method adjusts trade amounts in td so that
        all trades are expressed in terms of pnl_ccy. Trades with missing buy_ccy or sell_ccy are excluded.

        Parameters
        ----------
        td : DataFrame
            A DataFrame containing trade data. Expected columns are 'buy_ccy', 'sell_ccy', 'buy_amount',
            'sell_amount', and 'date'. Each row represents a trade with currencies and amounts for buying
            and selling, along with the trade date.
        fx : DataFrame
            A DataFrame containing FX rate data. Expected columns are 'base', 'quote', 'date', and 'rate'.
            The 'base' and 'quote' columns represent currency pairs, with 'rate' indicating the FX rate
            (quote / base) on the given 'date'.
        pnl_ccy : str, optional (default='GBP')
            The target profit and loss currency for standardizing trades.
            All trades in td are converted to this currency using FX rates from fx,
            if either leg is not already in the pnl_ccy.
        include_rate: bool, optional (default=False)
            if True include exchange 'rate' (buy_amount/sell_amount) and rate_source
        verbose : bool, optional (default=True)
            If True, prints additional information during processing.

        Returns
        -------
        DataFrame
            A DataFrame where each trade from td is expressed in pnl_ccy. The method adjusts trade amounts
            based on the corresponding FX rates in fx and splits trades not originally in pnl_ccy into
            separate trades. The output DataFrame retains the original columns from td.

        Notes
        -----
        - Trades in the input DataFrame (td) without both 'buy_ccy' and 'sell_ccy' are not processed.
        - If a suitable FX rate for a trade is not found, an error message is printed (if verbose is True),
          but the method continues processing other trades.
        - Assumptions:
            * FX rates are assumed to be triangulated, allowing conversion between any two currencies.
            * The method only uses fx where 'quote' == pnl_ccy. Any fx rates where 'quote' is different
              from pnl_ccy are ignored.

        Raises
        ------
        AssertionError
            If required columns are missing in either td or fx, or if there are null or zero-length
            strings in 'buy_ccy' or 'sell_ccy' in td.
        """

        # TODO: make sure error if missing any needed fx rates
        # TODO: re-write docstring - explain what is gong on
        # TODO: check run time for this function (?)
        # TODO: specify the output columns
        # TODO: explicitly define

        # required columns
        req_col_td = ["buy_ccy", "sell_ccy", "buy_amount", "sell_amount", "date"]
        # required fx columns
        req_col_fx = ['base', 'quote', 'date', 'rate']

        # check required columns are in DataFrames
        for k, v in {"td": [req_col_td, td], "fx": [req_col_fx, fx]}.items():
            missing = [c for c in v[0] if c not in v[1]]
            assert len(missing) == 0, f"{k}: missing columns: {missing}"

        # required buy/sell ccy is never null, or len 0
        for bs in ['buy_ccy', 'sell_ccy']:
            _ = pd.isnull(td[bs])
            assert not _.any(), f"{bs} has null\n{td.loc[_]}"
            _ = td[bs].map(lambda x: len(x)) == 0
            assert not _.any(), f"{bs} has len 0 entries\n{td.loc[_]}"

        if verbose:
            # print("-" * 100)
            print('splitting trades to be against: %s' % pnl_ccy)
            print(f"taking only fx rate where quote == {pnl_ccy}")

        # take fx where quote (numerator in rate) is the pnl_ccy
        # TODO: confirm fx rates are / can be inverted when processed after reading in
        fx = fx.loc[fx['quote'] == pnl_ccy].copy(True)

        assert len(fx), "no fx data found when matching 'quote' == pnl_ccy"

        # get all the base ccy, given quote is the pnl ccy
        fx_base = fx['base'].unique()

        # 'reset' index
        # TODO: why? remove
        # td.index = np.arange(len(td))

        # double book - any trade that is not in target ccy, split the trade in 2
        if verbose:
            print(f'splitting trades not in target pnl currency {pnl_ccy} into two trades')

        # get the unique trading pairs (direction matters here)
        td_pairs = td[['buy_ccy', 'sell_ccy']].drop_duplicates()

        # trade df should retain their columns
        org_cols = td.columns

        if include_rate:
            org_cols += ['rate', 'rate_source']

        # TODO: change trade split to be by trading pair - is this needed?
        # increment over each trade pair
        l = []
        for idx, row in td_pairs.iterrows():

            buy_ccy, sell_ccy = row['buy_ccy'], row['sell_ccy']

            # extract all trades matching this buy/sell pair
            df = td.loc[(td['buy_ccy'] == buy_ccy) & (td['sell_ccy'] == sell_ccy)].copy(True)

            # check if pnl ccy is in either the buy or sell ccy
            if (sell_ccy == pnl_ccy) | (buy_ccy == pnl_ccy):
                if include_rate:
                    df['rate'] = df['buy_amount'] / df['sell_amount']
                    df['rate_source'] = "implied"
                else:
                    l += [df]

            # otherwise need to split the trade
            else:
                # if the sell_ccy can be found as a base then use that one
                # - it's assumed fx rates are triangulated and it shouldn't matter if match on sell or buy ccy

                try:
                    if sell_ccy in fx_base:
                        bs, other = "sell", "buy"
                    elif buy_ccy in fx_base:
                        bs, other = "buy", "sell"
                    else:
                        raise NotImplementedError(f'could not find fx rate to split trade! ' 
                                                  f'buy_ccy: {buy_ccy}, sell_ccy: {sell_ccy}\ntrades:\n{df.head()}')

                    # merge on the fx rate
                    tmp = df.merge(fx,
                                   left_on=['date', f'{bs}_ccy'],
                                   right_on=['date', 'base'], how='left')

                    # check if any
                    missing_tmp = tmp.loc[pd.isnull(tmp['rate'])]

                    assert len(missing_tmp) == 0, (f"merging fx rate data onto:\n{row}\n"
                                                   f"had {len(missing_tmp)} entries missing rate. "
                                                   f"here's the tail\n{missing_tmp.tail(10)}")

                # if triggered assertion error try 'the other way'
                except AssertionError as e:

                    bs, other = ("buy", "sell") if bs == 'sell' else ("sell", "buy")

                    # merge on the fx rate
                    tmp = df.merge(fx,
                                   left_on=['date', f'{bs}_ccy'],
                                   right_on=['date', 'base'], how='left')

                    # check if any
                    missing_tmp = tmp.loc[pd.isnull(tmp['rate'])]

                    assert len(missing_tmp) == 0, (f"merging fx rate data onto:\n{row}\n"
                                                   f"had {len(missing_tmp)} entries missing rate. "
                                                   f"here's the tail\n{missing_tmp.tail(10)}")


                # split the trade
                t1, t2 = tmp.copy(True), tmp.copy(True)

                # leg 1: update other amount/ccy to be in pnl ccy amount
                # - pnl_ccy amount calculated from (units): bs_amount (bs_ccy) * rate (pnl_ccy / bs_ccy)
                t1[f'{other}_amount'] = t1[f'{bs}_amount'] * t1['rate']
                t1[f'{other}_ccy'] = pnl_ccy

                # leg 2: update bs amount/ccy to be equal to leg 1 other amount/ccy
                t2[f'{bs}_amount'] = t1[f'{other}_amount']
                t2[f'{bs}_ccy'] = pnl_ccy

                if include_rate:
                    # rate always given as buy/sell
                    t1['rate'] = t1['buy_amount'] / t1['sell_amount']
                    t2['rate'] = t2['buy_amount'] / t2['sell_amount']

                    t1.rename(columns={"source": "rate_source"}, inplace=True)
                    t2.rename(columns={"source": "rate_source"}, inplace=True)

                # keep only original trade columns
                t1, t2 = t1[org_cols], t2[org_cols]

                # add result to - faster to use extend()?
                l += [t1, t2]

        # combine the results
        df = pd.concat(l)

        # require buy/sell amounts are never None/nan
        nan_buy_amount = df.loc[pd.isnull(df['buy_amount'])]
        nan_sell_amount = df.loc[pd.isnull(df['sell_amount'])]

        assert len(nan_buy_amount) == 0, f"found null values in 'buy_amount'\n{nan_buy_amount}"
        assert len(nan_sell_amount) == 0, f"found null values in 'sell_amount'\n{nan_sell_amount}"

        return df

    @staticmethod
    def same_day_trades(df, threshold=1e-10):
        """
        Identify and categorize trades based on the same day rule.

        This method processes a DataFrame of trades, adding a column to indicate whether each transaction
        (either in full or partially) falls under the same day rule.


        Parameters
        ----------
        df : DataFrame
            A DataFrame containing transaction data for a single asset type (such as shares or currency).
            Expected columns include 'date', 'buy/sell', 'buy_amount', 'sell_amount', and 'rate'.
        threshold : float, default 1e-10
            trades with buy_amount and sell_amount below threshold are dropped

        Returns
        -------
        DataFrame
            A modified DataFrame with added columns 'rule' and 'rule_key'. The 'rule' column indicates the
            tax rule applicable to the transaction (Same Day, B&B, Section 104, or ''). The 'rule_key' column
            links offsetting trades.

        Raises
        ------
        AssertionError
            If there are more than two aggregated trades on any given day or if the net amounts change post-processing.

        Notes
        -----
        - The method splits trades occurring on the same day into separate entries, adjusting the amounts to reflect same day rules.
        - It adds two new columns to the DataFrame: 'rule' for the type of rule applicable, and 'rule_key' to link related trades.
        - Assumption: The function assumes that the input DataFrame is well-structured with necessary columns and pertains to transactions of a single asset type.


        """

        # TODO: confirm trades are expected to be split already
        # TODO: clean up/refactor duplicate code below

        # add a column indicating the rule the transaction fall under:
        # - Same Day, B&B, Section 104 (=='')
        # TODO: default rule should be 0,
        df['rule'] = ''

        # add a key / detail column, so the trades that are off setting can be linked
        df['rule_key'] = ''

        # get all the days with buy and sell on the same day
        sd = pd.pivot_table(df, index='date', values='buy/sell', aggfunc='count').reset_index()

        # sanity check: there should only be at most two (aggregated) trades on a day
        assert (sd['buy/sell'] <= 2).all(), sd[sd['buy/sell'] > 2]

        # select the dates that only have 1 trade - these will be returned as is
        x = df[df['date'].isin(sd[sd['buy/sell'] == 1].date)].copy()

        # get the two-day trades
        y = df[df['date'].isin(sd[sd['buy/sell'] == 2].date)].copy()

        # get the dates
        chk_dates = np.unique(y['date'].values)

        # if there are no dates with two trades per day, just return the original data
        if len(chk_dates) == 0:
            return df

        # else there are any days with two trades per day - split up the trades
        else:
            # store results in a list
            res = []

            # increment over each date, split the buy / sell amounts
            for d in chk_dates:
                # select data for date
                tmp = y[y['date'] == d].copy()
                # split the buy / sell amount
                s = tmp[tmp['buy/sell'] == 'sell'].copy(True)
                b = tmp[tmp['buy/sell'] == 'buy'].copy(True)

                # double check buy and sell amounts totals for day only have 1 row each
                assert len(s) == 1, f"sell amount for date: {d} expected to be aggregated, got: {s}"
                assert len(b) == 1, f"buy amount for date: {d} expected to be aggregated, got: {b}"

                # create a rule key
                rk = pd.to_datetime(d).strftime('%Y%m%d') + '_1'

                # if the sell amount is less than the buy amount
                # - then split the buy amount
                sell_amt = s['sell_amount'].values[0]
                buy_amt = b['buy_amount'].values[0]
                if sell_amt <= buy_amt:
                    # get the remaining (buy) amount
                    b_remain = buy_amt - sell_amt

                    # split the trade, add descriptions
                    b1, b2 = b.copy(True), b.copy(True)

                    # b1 covers the amount bought that day equal to the amount sold
                    # - b2 covers the remainder (extra) bought
                    b1['buy_amount'] = sell_amt
                    b2['buy_amount'] = b_remain

                    # update the sell amounts - using the rate
                    b1['sell_amount'] = b1['buy_amount'] * b1['rate']
                    b2['sell_amount'] = b2['buy_amount'] * b2['rate']

                    b1['rule'] = 1
                    b1['rule_key'] = rk

                    s['rule'] = 1
                    s['rule_key'] = rk

                    # concat the split trades
                    b = pd.concat([b1, b2])

                # otherwise the sell amount is more than buy amount
                # - so split the sell amount
                else:

                    # extra sold
                    s_remain = sell_amt - buy_amt
                    # split the trade, add descriptions
                    s1, s2 = s.copy(True), s.copy(True)

                    # s1 covers the amount sold that day equals to the amount bought
                    # - s2 covers the remainder (extra) sold
                    s1['sell_amount'] = buy_amt
                    s2['sell_amount'] = s_remain

                    # update the buy amounts
                    s1['buy_amount'] = s1['sell_amount'] * s1['rate']
                    s2['buy_amount'] = s2['sell_amount'] * s2['rate']

                    # add rule and key numbers
                    s1['rule'] = 1
                    s1['rule_key'] = rk

                    b['rule'] = 1
                    b['rule_key'] = rk

                    # concat the split trades
                    s = pd.concat([s1, s2])

                res += [pd.concat([b, s])]

            # combine the results
            # - dates that didn't require any splits and those that did
            out = pd.concat([x, pd.concat(res)])

            # sort by date
            out.sort_values('date', ascending=True, inplace=True)

            # check the net amounts did not change
            chk1 = pd.pivot_table(out, index=['date', 'buy/sell'], values=['buy_amount', 'sell_amount'],
                                  aggfunc='sum')
            chk2 = pd.pivot_table(df, index=['date', 'buy/sell'], values=['buy_amount', 'sell_amount'],
                                  aggfunc='sum')

            # check that the amounts add back up
            assert not (np.absolute(chk1 - chk2).values.sum() > 1e-6).any(), \
                'the amount bought/sold does not add up after it was split for same day calc: %.9f' % \
                np.absolute(chk1 - chk2).values.sum()

            # drop trades wit
            below_thresh = (out['sell_amount'] < threshold) & (out['buy_amount'] < threshold)

            if below_thresh.any():
                # drop those below threshold - this was added to for trades that exactly net
                out = out.loc[~below_thresh].copy(True)

            return out

    @staticmethod
    def bed_and_breakfast(df, hold_period=30):
        """
        Identify bed and breakfast trades within a DataFrame,
        which are defined as, for a given asset, a sell followed by a buy within a given number of days (default 30).

        This method examines a DataFrame of trades and identifies bed and breakfast trades,
        where an asset is sold and then bought back within a specified hold period (default 30 days).
        It categorizes these trades and adjusts the transaction amounts accordingly.

        Parameters
        ----------
        df : DataFrame
            A DataFrame containing financial transaction data. Expected to have 'date', 'buy/sell', 'buy_amount', 'sell_amount', 'rate', and 'rule' columns.
        hold_period : int, optional
            The number of days within which a buy transaction following a sell transaction is considered a bed and breakfast trade. Default is 30 days.

        Returns
        -------
        DataFrame
            A modified DataFrame with transactions categorized as bed and breakfast trades, where applicable. The DataFrame includes updated 'buy_amount' and 'sell_amount' for these trades.

        Raises
        ------
        AssertionError
            If the aggregated buy/sell amounts do not match the original totals post-processing.

        Notes
        -----
        - The function first excludes trades accounted for by the 'same day' rule.
        - It then iterates over the remaining transactions to identify and process bed and breakfast trades.
        - Assumption: The input DataFrame is structured with necessary transaction details, and 'same day' trades are already marked with a rule identifier.

        """
        # TODO: this is pretty messy and could probably be cleaned up
        # store the original dataframe to check against the the resulting output
        org = df.copy()

        # remove the trades which have been account for using the 'same day' rule (1)
        same_day = df[df['rule'] == 1].copy()

        # get the rest
        df = df[df['rule'] != 1].copy(True)

        # reset the index - because will use index to select over dates
        df.index = np.arange(len(df))

        # store 30 day (bed and breakfast) trades in list
        off_sets = []

        # TODO: because bellow uses an index = np.arrange(), don't need to use iloc, could/should just use loc

        # increment over each date, check if sales were bought back within 30 days
        # - index is only up to second last because
        # - for the last trade there are no further trades to check if bought or sold back
        for i in range(len(df)-1):
            # get the trade
            trd = df.iloc[i]

            # if the trade is a sell, look at the following days to see if there is a buy back
            if trd['buy/sell'] == 'sell':

                # get the sell amount - to offset if need be
                sa = trd['sell_amount'].copy()

                # get the rule key (if needed)
                rk = trd['date'].strftime('%Y%m%d') + '_2'

                # increment over the following days
                for j in range(i + 1, len(df)):
                    # if date is more than 30 days break
                    if (df.loc[j, 'date'] - trd.date).days > hold_period:
                        break

                    # if the sell amount is zero, then stop
                    # - it could be zero if reduced by offsets
                    if sa == 0:
                        break

                    # if less than 30 days, check if buy and if buy amount > 0
                    # (buy amounts can be reduced to zero from offsets)
                    if (df.loc[j, 'buy/sell'] == 'buy') & (df.loc[j, 'buy_amount'] > 0):
                        # if the buy amount is less than the sell amount
                        # - then the entire buy amount will be offset
                        if df.loc[j, 'buy_amount'] < sa:
                            # copy the buy amount
                            b = df.iloc[[j]].copy()
                            b['rule'] = 2
                            b['rule_key'] = rk
                            # store the offset buy trade into a list
                            off_sets += [b]
                            # update the buy/sell amounts to be zero (as it's been entirely accounted for)
                            df.loc[j, 'buy_amount'], df.loc[j, 'sell_amount'] = 0, 0
                            # reduce the sell amount
                            sa = sa - b['buy_amount'].values[0]
                        # otherwise the buy amount covers the sell amount
                        else:
                            # copy the buy amount
                            b = df.iloc[[j]].copy()
                            b['rule'] = 2
                            b['rule_key'] = rk
                            # the buy amount will be set to the sell amount, then update the sell amount
                            b['buy_amount'] = sa
                            b['sell_amount'] = b['buy_amount'] * b['rate']
                            # store the offset buy trade
                            off_sets += [b]
                            # update (reduce) the remaining buy/sell amount for the trade
                            df.loc[j, 'buy_amount'] = df.loc[j, 'buy_amount'] - b['buy_amount'].values[0]
                            df.loc[j, 'sell_amount'] = df.loc[j, 'sell_amount'] - b['sell_amount'].values[0]
                            # the sell amount is then entirely covered and should be set zero
                            sa = 0

                # if the sell amount as been offset (i.e. is different from starting point)
                # - then separate out the offset amount
                if sa != trd['sell_amount']:
                    # copy the sell trade - to offset
                    s = df.iloc[[i]].copy()
                    s['rule'] = 2
                    s['rule_key'] = rk
                    # update the sell amount - reduce by the amount that has been offset
                    s['sell_amount'] = s['sell_amount'] - sa
                    # update the buy_amount as well
                    s['buy_amount'] = s['sell_amount'] * s['rate']
                    # store the offset sell trade
                    off_sets += [s]

                    # update (reduce) the remaining buy/sell amount for the trade
                    df.loc[i, 'sell_amount'] = sa
                    df.loc[i, 'buy_amount'] = df.loc[i, 'sell_amount'] * df.loc[i, 'rate']

        # combine results
        # remove trades with zero amount
        df = df[df['buy_amount'] > 0]

        # add back in the same day
        df = pd.concat([df, same_day])

        # if there were any off_set trades, add them back in
        if len(off_sets) > 0:
            df = pd.concat([df] + off_sets)

        # sort by date
        df.sort_values('date', ascending=True, inplace=True)

        # ----------------
        # check that the buy/sell amounts on each day are the same as when started (sanity check)
        chk1 = pd.pivot_table(df, index=['date', 'buy/sell'], values=['buy_amount', 'sell_amount'],
                              aggfunc='sum')  # .reset_index()
        chk2 = pd.pivot_table(org, index=['date', 'buy/sell'], values=['buy_amount', 'sell_amount'],
                              aggfunc='sum')  # .reset_index()

        # check that the amounts add back up (within machine error or so)
        assert not (np.absolute(chk1 - chk2).values.sum() > 1e-9).any(), \
            'the amount bought/sold does not add up after it was split for same day calc: %.9f' % \
            np.absolute(chk1 - chk2).values.sum()
        # ----------------

        return df

    @staticmethod
    def average_price(p, amt, tol=-1e-10):
        """
        Calculate the average price of a financial asset, based on transaction prices and amounts.

        This method computes the average price per unit of an asset over a series of transactions.
        It is designed for use in scenarios where the asset is bought in multiple transactions, possibly at different prices.
        The method does not account for short selling and assumes all transaction amounts to be positive.
        A warning is raised if the net position becomes negative (indicating short selling) or if the first transaction is a sale.

        Parameters
        ----------
        p : ndarray
            An array of prices at which the transactions occurred.
        amt : ndarray
            An array of transaction amounts. Positive values indicate purchases, and negative values indicate sales.
        tol : float, optional
            The tolerance level below which the net position is considered negative, indicating potential short selling due to rounding errors. Default is -1e-10.

        Returns
        -------
        ndarray
            An array of the same length as `p` and `amt`, containing the average price per unit after each transaction.

        Raises
        ------
        Warning
            If the net position becomes negative or if the first transaction is a sale.

        Notes
        -----
        - The average price is recalculated after each transaction, remaining constant for sales and updating for purchases.
        - Assumption: The first transaction is expected to be a purchase, and subsequent transactions can be either purchases or sales.

        Examples
        --------
        >>> prices = np.array([100, 102, 101, 103])
        >>> amounts = np.array([10, 5, -3, 7])
        >>> average_prices = CapitalGains.average_price(prices, amounts)
        >>> print(average_prices)


        """
        # get the net position
        pos = np.cumsum(amt)

        # add checks on the data
        # assert amt[0] > 0, 'assumed first transaction is a purchase'
        # assert (pos >= 0).all(), 'net position goes negative, short selling not handled'
        if not (pos >= tol).all():
            warnings.warn('net position goes negative')
        if not (amt[0] > tol):
            warnings.warn('first transaction is a sale')

        # initialise an array for the average price
        p_ = np.full(len(p), np.nan)

        # set the first value - assumes the first element is a buy!
        p_[0] = p[0]

        # increment over the remaining transactions
        for i in range(1, len(p)):
            # if transaction was a sell, average price remains the same
            if amt[i] < 0:
                p_[i] = p_[i - 1]
            # otherwise a buy, average price needs to be updated
            else:
                # the total position - position from previous plus new amount
                w = pos[i - 1] + amt[i]
                # weight the previous average price with the new price
                p_[i] = p_[i - 1] * (pos[i - 1] / w) + p[i] * (amt[i] / w)

        # return the average price
        return p_

    @classmethod
    def get_pnl_by_rules(cls, df, pnl_ccy):
        """
        Calculate profit and loss (PnL) by applying different tax rules to a dataset of financial trades.

        This method processes a DataFrame of trades, applying specific tax rules such as same day, bed and breakfast (30-day rule), and Section 104 holding to calculate PnL for each trade. It categorizes trades based on these rules and computes the PnL accordingly.

        Parameters
        ----------
        df : DataFrame
            A DataFrame containing trade data. Expected to have columns 'date', 'buy_ccy', 'sell_ccy', 'buy/sell', 'buy_amount', 'sell_amount', and 'rule'.
        pnl_ccy : str
            The currency in which PnL should be calculated.

        Returns
        -------
        DataFrame
            A DataFrame with the same structure as `df`, enriched with PnL calculations and additional information like 'average_price' for each trade.

        Notes
        -----
        - The method identifies trades falling under same day rule, bed and breakfast rule, and Section 104 holding, and calculates the PnL for each category.
        - Assumption: Trades are already categorized with a 'rule' field, and the method expects the dataset to have necessary fields for PnL calculation.
        - The method depends on the `average_price` class method for calculating the average price of trades under Section 104 holding.

        """
        # TODO: clean up duplicate code below!

        # -------------------
        # same day pnl
        # -------------------
        sd = df[df['rule'] == 1].copy()

        # increment overrule keys, store results in a list
        l = []
        for rk in sd['rule_key'].unique():
            # get the elements matching rule_key
            tmp = sd[sd['rule_key'] == rk]
            # get the amount sold for (in GBP), subtract the amount bough for (GBP)
            pnl = tmp.loc[tmp['buy/sell'] == 'sell', 'buy_amount'].sum() - \
                  tmp.loc[tmp['buy/sell'] == 'buy', 'sell_amount'].sum()
            # put the results in dataframe
            l += [pd.DataFrame({'rule_key': rk, 'pnl': pnl, 'buy/sell': 'sell'}, index=[0])]

        # if there were any entries merge them on
        if len(l) > 0:
            # combine the results and merge onto same day data
            same_day_pnl = pd.concat(l)

            sd = sd.merge(same_day_pnl, on=['rule_key', 'buy/sell'], how='left')
            sd.fillna(0, inplace=True)

        # -------------------
        # 30 day rule: 2
        # -------------------

        bb = df[df['rule'] == 2].copy()

        # increment overrule keys, store results in a list
        l = []
        for rk in bb['rule_key'].unique():
            # get the elements matching rule_key
            tmp = bb[bb['rule_key'] == rk]
            # get the amount sold for (in GBP), subtract the amount bough for (GBP)
            pnl = tmp.loc[tmp['buy/sell'] == 'sell', 'buy_amount'].sum() - \
                  tmp.loc[tmp['buy/sell'] == 'buy', 'sell_amount'].sum()
            # put the results in data.frame
            l += [pd.DataFrame({'rule_key': rk, 'pnl': pnl, 'buy/sell': 'sell'}, index=[0])]

        # if there were any entries merge them on
        if len(l) > 0:
            # combine the results and merge onto same day data
            bnb_pnl = pd.concat(l)

            bb = bb.merge(bnb_pnl, on=['rule_key', 'buy/sell'], how='left')
            bb.fillna(0, inplace=True)

        # -------------------
        #  section 104 - rule: ''
        # -------------------

        # TODO: what if there is none - this should be done better
        s1 = df[df.rule == ''].copy()

        # HACK: to handle if there are no s104 rules
        if len(s1) > 0:

            pnl_ccy_amount = f'amount_{pnl_ccy}'

            # get the ccy and pnl ccy (GBP) amounts
            # TODO: should match on ccy type, not the implied
            s1[pnl_ccy_amount] = np.where(s1['buy/sell'] == 'buy',
                                          s1['sell_amount'],
                                          s1['buy_amount'])
            s1['amount_CCY'] = np.where(s1['buy/sell'] == 'buy',
                                        s1['buy_amount'],
                                        s1['sell_amount'])

            # price (duplication of rate)
            s1['p'] = s1[f'amount_{pnl_ccy}'] / s1['amount_CCY']

            # get the signed position
            amt = np.where(s1['buy/sell'] == 'buy', s1['amount_CCY'], -s1['amount_CCY'])

            # price
            p = s1['p'].values

            # TODO: double check this calculation!
            # average price calculation
            # - done incrementally by considering previous amount and price new element purchased for
            ap = cls.average_price(p, amt)

            s1['average_price'] = ap

            # get the pnl
            # - pnl = 0 if buying, if selling pnl = amount in pnl_ccy - (average price * amount sold)
            s1['pnl'] = np.where(s1['buy/sell'] == 'buy', np.zeros(len(s1)),
                                 s1[pnl_ccy_amount] - (s1['amount_CCY'] * s1['average_price']))

            s104 = df[df.rule == ''].merge(s1[['date', 'buy_ccy', 'sell_ccy', 'buy/sell', 'average_price', 'pnl']],
                                           on=['date', 'buy_ccy', 'sell_ccy', 'buy/sell'], how='left')

            out = pd.concat([sd, bb, s104], sort=False)
        else:
            out = pd.concat([sd, bb], sort=False)

        # add average price column - for the event section 104 wasn't applied
        if 'average_price' not in out:
            out['average_price'] = np.nan

        # sort by date
        out.sort_values('date', inplace=True, ascending=True)

        return out

    @staticmethod
    def find_tax_yr(date, tax_year_start=None):
        """
        Determine the tax year for a given date. The tax year is the calendar year the tax year ends.

        This method calculates the tax year based on a provided date.
        In the UK, the tax year runs from April 6th to April 5th of the following year.

        Parameters
        ----------
        date : str or datetime-like
            The date for which the tax year needs to be determined.
            Can be a string in a date-format or a datetime-like object.
            If string it is converted to datetime via pd.to_datetime
        tax_year_start : str or None, default None


        Returns
        -------
        int
            The calendar year in which the tax year, associated with the given date, ends.

        Notes
        -----
        - The method assumes that if the date is after April 5th, it falls into the tax year ending in the following calendar year.
        - Assumption: The tax year is defined as starting from April 6th of one year to April 5th of the next year. This aligns with the UK tax year but may differ in other jurisdictions.

        Examples
        --------
        >>> date = "2023-04-06"
        >>> tax_year = CapitalGains.find_tax_yr(date)
        >>> print(tax_year)  # Output: 2024
        """

        if isinstance(date, (float, int, type(None))):
            raise TypeError("type(date): {type(date)} not handled")

        if tax_year_start is None:
            tax_year_start = "04-06"
            # print(f"tax_year_start not supplied, using: '{tax_year_start}'")

        # tax year: the last calendar year of the tax year
        # get calendar year of date
        date = pd.to_datetime(date)

        y = date.strftime("%Y")

        # if is after april 5 (?) then in 'next' tax year, e.g. the one ending in the following year
        if date >= np.datetime64(f"{y}-{tax_year_start}"):
            return int(y) + 1
        else:
            return int(y)

    @staticmethod
    def _sum_pnl_on_index(df, index):
        """
        Summarize profit and loss (PnL) and other financial metrics for sales within a tax year,
        grouped by a specified index.

        This method processes a DataFrame containing sales data and calculates aggregated
        financial metrics such as total buy amount, allowable cost, and PnL.
        The aggregation is performed based on specified index(es).
        It also counts the number of disposals for each group.

        Parameters
        ----------
        df : DataFrame
            A DataFrame containing sales data. Expected to include 'buy_amount', 'allowable_cost', 'pnl', and 'buy_ccy' columns.
        index : str or list
            The column name(s) on which to group the data before aggregating. Can be a single column name or a list of column names.

        Returns
        -------
        DataFrame
            A DataFrame with aggregated financial metrics and disposal counts, grouped by the specified index.

        Raises
        ------
        AssertionError
            If the `index` is not a list or if the DataFrame contains more than one unique 'buy_ccy'.

        Notes
        -----
        - The method is designed for use with data where all transactions are in the same 'buy_ccy'.
        - Assumption: The DataFrame `df` is structured with necessary financial transaction details, and the aggregation is meaningful for the provided `index`.
        """

        # sum: buy_amount, allowable_cost and pnl
        # count date

        # want index to be list
        if isinstance(index, str):
            index = [index]

        assert isinstance(index, list), f"index is not a list, it is: {type(index)}"

        # assert there is only one buy_ccy (say pnl_ccy)
        assert len(df['buy_ccy'].unique()) == 1, \
            f"expecting one buy_ccy, got: {df['buy_ccy'].unique()}"

        # check each index is dataframe
        for i in index:
            assert i in df, f"index element: {i} was not in df columns: {df.columns}"

        sum_pnl = pd.pivot_table(df,
                                 index=index,
                                 values=['buy_amount', 'allowable_cost', 'pnl'],
                                 aggfunc='sum')

        # count the number of disposables per ccy
        disposal_count = pd.pivot_table(df,
                                        index=index,
                                        values=['date'],
                                        aggfunc='count')

        # combine to get detailed summary, rename columns, sort by P&L
        det_summary = pd.concat([sum_pnl, disposal_count], axis=1)
        det_summary.rename(columns={'buy_amount': 'Proceeds',
                                    'date': 'Num Disposals',
                                    'allowable_cost': 'Allowable Costs',
                                    'pnl': 'P&L'}, inplace=True)
        det_summary.sort_values("P&L", inplace=True, ascending=False)

        return det_summary

    @classmethod
    def get_summary_for_ty_ccy(cls, ty, pnl_ccy, extra_index=None):
        """
        Generate a summary of tax year transactions for a specified profit and loss currency (pnl_ccy).

        This method provides detailed summaries of sales transactions within a tax year, filtered by a specified pnl currency. It calculates allowable costs and aggregates data to create comprehensive summaries at different levels: detailed per currency and listed status, overall per listed status, and a detailed breakdown of each sale.

        Parameters
        ----------
        ty : DataFrame
            A DataFrame containing tax year transaction data. Expected to include 'buy_amount', 'pnl', 'buy_ccy', 'sell_ccy', and 'listed' columns.
        pnl_ccy : str
            The currency in which profits and losses are calculated and summarized.

        Returns
        -------
        tuple of DataFrames
            A tuple containing three DataFrames:
            - Summary per listed status.
            - Detailed summary per 'sell_ccy' and 'listed' status.
            - Detailed summary breakdown for each sale in the tax year.

        Raises
        ------
        AssertionError
            If not all transactions in the filtered DataFrame are sales ('buy/sell' column values are not all 'sell').

        Notes
        -----
        - The method assumes that all transactions to be summarized are sales and are filtered by the pnl currency.
        - Assumption: The input DataFrame `ty` is structured with the necessary financial transaction details for accurate summarization.

        """
        # TODO: provide aggregation levels, via a list of index columns

        if extra_index is None:
            extra_index = []

        # sales in tax year
        ty_sales = ty.loc[ty['buy_ccy'] == pnl_ccy].copy(True)

        # expect buy/sell column to all be sold, this is perhaps not a useful check
        assert (ty_sales['buy/sell'] == 'sell').all(), \
            f"not all values in buy/sell column are 'sell' when selecting those with buy_ccy == {pnl_ccy}:" \
            f"\n{ty_sales.loc[ty_sales['buy/sell'] != 'sell']}"

        # allowable cost is the buy amount (in pnl ccy) minus the pnl
        # - assumes pnl calc is correct!
        ty_sales['allowable_cost'] = ty_sales['buy_amount'] - ty_sales['pnl']

        # detailed summary - break down per ccy
        det_summary = cls._sum_pnl_on_index(df=ty_sales, index=['sell_ccy', 'tax_year'])

        # over all summary
        summary = cls._sum_pnl_on_index(df=ty_sales, index=['tax_year'])

        # detailed_summary break down - each sale of the year
        det_sum_breakdown = ty_sales.copy(True)

        return summary, det_summary, det_sum_breakdown

    def aggregate_daily(self, df, pnl_ccy="GBP"):

        # TODO: add checks on columns

        # require trades are all expressed against pnl_ccy (either buy or sell)
        has_pnl_ccy = (df['sell_ccy'] == pnl_ccy) | (df['buy_ccy'] == pnl_ccy)
        assert has_pnl_ccy.all(), f"not all trades\n{df.loc[~has_pnl_ccy]}\nare against pnl_ccy={pnl_ccy}, can't pool"

        # get the daily total amounts, by ccy pair
        # - this keeps buys and sells separate (rows)
        tot = pd.pivot_table(df,
                             index=['date', 'buy_ccy', 'sell_ccy'],
                             values=['sell_amount', 'buy_amount'],
                             aggfunc='sum').reset_index()

        # get the exchanges - combining multiple where needed
        # tot_ex = pd.pivot_table(df,
        #                         index=['date', 'buy_ccy', 'sell_ccy'],
        #                         values=['exchange'],
        #                         aggfunc=lambda x: "|".join(np.unique(x).tolist())).reset_index()
        #
        # tot = tot.merge(tot_ex, on=['date', 'buy_ccy', 'sell_ccy'], how='left')

        # add a 'rate', units of pnl_ccy / other_ccy
        # - used in same_day_trades
        tot['rate'] = np.where(tot['sell_ccy'] == pnl_ccy,
                               tot['sell_amount'] / tot['buy_amount'],
                               tot['buy_amount'] / tot['sell_amount'])

        # 'buy/sell' column need for same_day_trades, bed_and_breakfast (?)
        tot['buy/sell'] = np.where(tot['sell_ccy'] == pnl_ccy, "buy", "sell")

        return tot

    def apply_rules_by_asset(self, tot, pnl_ccy="GBP",
                             bnb_hold_period=30,
                             same_day_threshold=1e-10):

        # get in ccy, which is not the pnl_ccy
        all_ccy = np.unique(np.concatenate([tot['sell_ccy'].unique(), tot['buy_ccy'].unique()]))
        all_ccy = all_ccy[~np.in1d(all_ccy, pnl_ccy)]

        ag = {}
        for ccy in all_ccy:
            # select data for ccy
            _ = tot.loc[(tot['sell_ccy'] == ccy) | (tot['buy_ccy'] == ccy)].copy(True)

            # apply trading rules
            # - same day
            sdt = self.same_day_trades(_, threshold=same_day_threshold)
            # - B&B
            ag[ccy] = self.bed_and_breakfast(sdt, hold_period=bnb_hold_period)

        return ag

    @staticmethod
    def _get_initials(s, split=" "):
        # only get initials is there is at least on split, otherwise return as
        _ = s.split(split)
        if len(_) > 1:
            out = "".join([i[0] for i in _ if len(i) > 0])
        else:
            out = s
        return out

    @classmethod
    def _add_aggregate_id(cls, s, tot, pnl_ccy):
        # TODO: add unit test

        all_ccy = np.unique(
            np.concatenate([tot['buy_ccy'].unique(), tot['sell_ccy'].unique()])
        )

        # require the inverse is the same size
        # all_ccy_map = {k: cls._get_initials(k, split=" ") for k in all_ccy}
        # with change in ii trade statements shortening longer names not really needed
        all_ccy_map = {k:k for k in all_ccy}

        invert_map = {v:k for k, v in all_ccy_map.items()}
        assert len(all_ccy_map) == len(invert_map), ("taking first letter of ccy name did not give 1:1 mapping,"
                                                     f"expected: {len(all_ccy_map)}, got: {len(invert_map)}")

        tot['agg_id'] = (# + "_" +
                         np.where(tot['buy_ccy'] == pnl_ccy,
                                  tot['sell_ccy'].map(all_ccy_map),
                                  tot['buy_ccy'].map(all_ccy_map)) + "_" +
                          tot['date'].dt.strftime('%Y%m%d') + "_" +
                         tot["buy/sell"])

        # merge on the aggregate id onto the trades
        tmp = []
        for bs in ["buy", "sell"]:
            _ = s.loc[s[f'{bs}_ccy'] != pnl_ccy].merge(tot[["date", f'{bs}_ccy', "agg_id"]],
                                                       on=['date', f'{bs}_ccy'],
                                                       how="left")
            tmp.append(_)

        assert sum([len(i) for i in tmp]) == len(s)

        s = pd.concat(tmp)

        return s

    def run(self, td, fx, pnl_ccy='GBP', bnb_hold_period=30, amount_threshold=1e-10, verbose=False):
        """
        Calculate profit and loss (PnL) for holdings based on various tax rules.

        This method processes a dataset of trades and foreign exchange rates, calculates the PnL for each trade, and applies various tax rules such as same day, bed and breakfast, and Section 104 holding. The trades are first standardized against a specified pnl currency. The method then pools trades, applies the tax rules, and aggregates the data to provide a comprehensive PnL calculation for each holding.

        Parameters
        ----------
        td : DataFrame
            A DataFrame containing trade data. Expected to include details of each trade.
        fx : DataFrame
            A DataFrame containing foreign exchange rate data.
        pnl_ccy : str, optional
            The profit and loss currency. All trades will be standardized to this currency. Defaults to 'GBP'.
        bnb_hold_period : int, optional
            The holding period for the bed and breakfast rule. Default is 30 days.
        amount_threshold: float, default 1e-10
            Trades that have buy_amount AND sell_amount below same_day_threshold will be dropped.
        verbose: bool, default False

        Returns
        -------
        DataFrame
            A DataFrame with detailed PnL calculations for each trade, including tax rule applications, listed status, and tax year.
        DataFrame
            Split Trades

        Notes
        -----
        - The method assumes that the input DataFrames `td` and `fx` are properly formatted and contain necessary details for PnL calculations.
        - Trades are first split to be against the pnl currency, then pooled and processed through the tax rules.
        - Assumption: The method includes logic to identify listed trades based on a predefined list of exchanges (attribute `self.listed_exchanges`).
        """

        print("run(...) started")

        # require tradeid be unique
        assert np.unique(td['tradeid'].values, return_counts=True)[1].max() == 1, "tradeid must be unique, duplicates were found"

        # --
        # split trades to always be against pnl_ccy
        # --

        print("-" * 10)
        print("splitting trades")
        s = self.split_trades(td.copy(True), fx, pnl_ccy=pnl_ccy)

        # ---
        # aggregate trades by date, apply rules
        # ---

        print("-" * 10)
        print("aggregating daily bought and sold")
        tot = self.aggregate_daily(s, pnl_ccy=pnl_ccy)

        # add an aggregate id - to allow mapping back to individual trades
        s = self._add_aggregate_id(s, tot, pnl_ccy=pnl_ccy)

        # --
        # apply rules
        # --

        print("-" * 10)
        print("applying capital gains rules")
        ag = self.apply_rules_by_asset(tot, pnl_ccy=pnl_ccy, bnb_hold_period=bnb_hold_period,
                                       same_day_threshold=amount_threshold)

        # HARDCODED: column order
        col_order = ['date', 'buy/sell',
                     'buy_ccy', 'buy_amount', 'sell_ccy', 'sell_amount',
                     'rate', 'pnl', 'rule', 'rule_key', 'agg_id', 'average_price'
                     # 'exchange'
                     ]

        pnl = {}
        for k, v in ag.items():
            if verbose:
                print('getting pnl for: %s' % k)

            tmp = self.get_pnl_by_rules(v, pnl_ccy=pnl_ccy)
            pnl[k] = tmp[col_order]

            if verbose:
                print(pnl[k].pnl.sum())

        # combine all the pnls
        net = pd.concat([v for k, v in pnl.items()])

        # ---
        # add tax year
        # ---

        # add the tax year
        net['tax_year'] = [self.find_tax_yr(np.datetime64(i)) for i in net['date']]

        # ---
        # map rules to more descriptive values
        # ---

        rule_map = {1: 'SameDay', 2: 'B&B', '': 'pool'}
        net['rule'] = net['rule'].map(rule_map)

        print("run(...) finished")

        return net, s

    @staticmethod
    def format_float_col(x, float_format='%.2f'):

        for i, dt in enumerate(x.dtypes):
            if dt == 'float64':
                # HACK: being lazy
                if x.columns[i] == 'Pool Price':
                    x[x.columns[i]] = ['na' if np.isnan(j) else '%.4f' % j for j in x[x.columns[i]]]
                else:
                    x[x.columns[i]] = ['na' if np.isnan(j) else float_format % j for j in x[x.columns[i]]]
        return x

    @classmethod
    def write_results_to_tex(cls, net, summary, det_summary, det_sum_breakdown,
                             out_dir, out_file):
        # TODO: this needs to be clean up! just awful
        # TODO: the following a rushed mess, needs tidying

        print("-" * 50)
        print("writing summary and break down of tax calculation to tex file")

        # if not tex_file_only:
        #     assert os.path.exists(cls.pdflatex), f"pdflatex path: {cls.pdflatex} does not exist"

        # mapping dict to rename columns
        rename_cols = {
            "sell_ccy": "Sell Asset",
            "buy_ccy": "Buy Asset",
            "sell_amount": "Sell Amount",
            "buy_amount": "Buy Amount",
            "allowable_cost": "Allowable Costs",
            "pnl": "P&L",
            "average_price": "Pool Price",
            "rule": "Rule",
            "rule_key": "Rule Key",
            # "exchange": "Exchange",
            "date": "Date"
        }

        print('writing summary results to file in the follow directory')
        print(out_dir)

        # create output
        if not os.path.exists(out_dir):
            print('%s\ndoes not exist, creating' % out_dir)
            os.makedirs(out_dir)

        # initialise the document to write to
        geometry_options = {'margin': '1.25cm', 'landscape': True}
        doc = Document(geometry_options=geometry_options)
        # doc = Document('basic')
        doc.documentclass = Command(
            'documentclass',
            options=['11pt', 'landscape'],
            arguments=['article'],
        )

        # add packages
        doc.packages.append(Package('booktabs'))  # for \midrule, \toprule?
        doc.packages.append(Package('lscape'))  # landscape
        doc.packages.append(Package('float'))
        doc.packages.append(Package('longtable'))
        doc.packages.append(Package('array'))

        # to left align all longtables
        doc.append(NoEscape("\setlength{\LTleft}{0pt}"))

        # HACK: make copies, as will reset index and manipulate
        summary = summary.copy(True)
        det_summary = det_summary.copy(True)
        det_sum_breakdown = det_sum_breakdown.copy(True)

        summary.reset_index(inplace=True)
        det_summary.reset_index(inplace=True)
        det_sum_breakdown.reset_index(inplace=True)

        # write data to file
        write_dict = {
            "full_calc": net.rename(columns=rename_cols),
            "sale_summary": summary.rename(columns=rename_cols),
            "detailed_sale_summary": det_summary.rename(columns=rename_cols),
            "detailed_sale_breakdown": det_sum_breakdown.rename(columns=rename_cols)
        }
        for k, v in write_dict.items():
            v.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)

        # ---
        # high level summary
        # ---

        doc.append(NoEscape('\\subsection{Summary}'))

        # select summary for section, format floats, select subset of columns,
        # - add to document
        x = summary.copy(True)
        s_col = ['Num Disposals', 'Proceeds', 'Allowable Costs', 'P&L']
        # format numbers to the right
        x = cls.format_float_col(x, float_format='%.2f')
        # add to doc
        doc.append(NoEscape(dataframe_to_tex(x[s_col], float_format='%.2f')))

        # ---
        # summary break down
        # ---

        doc.append(NoEscape('\\subsection{Summary Breakdown}'))

        # select summary for section, sort by pnl, format floats, select subset of columns,
        # - add to document
        x = det_summary.copy(True)
        x.sort_values('P&L', ascending=False, inplace=True)

        x['Sell Asset'] = x['sell_ccy']
        sum_col = ['Sell Asset', 'Num Disposals', 'Proceeds', 'Allowable Costs', 'P&L']
        x = cls.format_float_col(x, float_format='%.2f')

        doc.append(NoEscape(dataframe_to_tex(x[sum_col], float_format='%.2f')))

        # ccy to increment over - for calculation breakdown
        section_ccy = x['sell_ccy'].values

        # ----
        # Gains before losses and losses in year
        # ----

        year_pnl = x['P&L'].astype(float)
        gains = year_pnl[year_pnl > 0].sum()
        losses = year_pnl[year_pnl < 0].sum()

        doc.append(NoEscape(f"\nTotal Gains in Year: {gains:.2f}\nTotal losses in Year: {losses:.2f}\n"))

        # ---
        # sale break down (detailed)
        # ---

        doc.append(NoEscape('\\subsection{Tax Year Sales}'))
        # x = det_sum_breakdown.loc[det_sum_breakdown['section'] == s, :].copy(True)
        x = det_sum_breakdown.copy(True)
        x.rename(columns=rename_cols, inplace=True)
        x.sort_values('Date', ascending=True, inplace=True)

        sum_col = ['Date', 'Sell Asset', 'Sell Amount', 'Buy Asset', 'Buy Amount', 'P&L', 'Rule', 'Pool Price']#,
                  # 'Exchange']
        doc.append(NoEscape(dataframe_to_tex(x[sum_col], float_format='%.2f')))

        # ---
        # Calculation Break down
        # ---

        doc.append(NoEscape('\\subsection{Sale Asset Calculations}'))
        # for each sell ccy get the break-down of the calculation
        # - i.e. all transactions with that currency
        for i in section_ccy:
            print(f'writing detailed output for: {i}')
            x = net.loc[(net['sell_ccy'] == i) | (net['buy_ccy'] == i)].copy(True)

            # rename columns
            x.rename(columns=rename_cols, inplace=True)

            # shorten unit names
            for jj in ['Sell Asset', 'Buy Asset']:
                x[jj] = ['%s ...' % z[:12] if len(z) > 12 else z for z in x[jj]]

            x.sort_values('Date', inplace=True)

            doc.append(NoEscape('\n\\textbf{%s}\n' % i))
            # select a subset of columns
            sum_col = (['Date', 'Sell Asset', 'Sell Amount', 'Buy Asset', 'Buy Amount', 'P&L', 'Rule', 'Pool Price'] +
                       # ['Exchange'] +
                       ['Rule Key'])
            # using more decimals for detailed calculation
            doc.append(NoEscape(
                dataframe_to_tex(x[sum_col], float_format='%.4f')
            ))

        # --
        # generate a tex file
        # --

        out_file_full = os.path.join(out_dir, out_file)
        doc.generate_tex(filepath=out_file_full)
        print(f"generated .tex file:\n{out_file_full}.tex\ncopy to OverLeaf to compile")


if __name__ == "__main__":

    pass
