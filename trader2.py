import math
from typing import Dict, List

import numpy as np
import pandas as pd

from datamodel import Order, OrderDepth, Symbol, Trade, TradingState, UserId


class Trader:
    PRODUCTS = [
    'AMETHYSTS',
    'STARFRUIT',
    'ORCHIDS',
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS' : 10000,
        'STARFRUIT' : 5000,
        'ORCHIDS': 1000,
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
        'ORCHIDS': 100,
    }

    prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    mid_prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    errors_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}
    forecasted_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": []}

    current_signal = {"AMETHYSTS": "", "STARFRUIT": "None", "ORCHIDS": "None"}


    def __init__(self) -> None:

        self.ema_prices = dict()
        for product in self.PRODUCTS:
            self.ema_prices[product] = None
        self.ema_param = 0.

        self.window_size = 81

        self.orchids_df = pd.DataFrame()

        self.current_pnl = dict()
        self.qt_traded = dict()
        self.pnl_tracker = dict()

        for product in self.PRODUCTS:
            self.current_pnl[product] = 0
            self.qt_traded[product] = 0
            self.pnl_tracker[product] = []


        self.backtest = False
        if not self.backtest:
            self.orchids_df = load_data()


    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    
    
    def get_order_book(self, product, state: TradingState):
        market_bids = list((state.order_depths[product].buy_orders).items())
        market_asks = list((state.order_depths[product].sell_orders).items())

        if len(market_bids) > 1:
            bid_price_1, bid_amount_1 = market_bids[0]
            bid_price_2, bid_amount_2 = market_bids[1]

        if len(market_asks) > 1:
            ask_price_1, ask_amount_1 = market_asks[0]
            ask_price_2, ask_amount_2 = market_asks[1]


        bid_price, ask_price = bid_price_1, ask_price_1

        if bid_amount_1 < 5:
            bid_price = bid_price_2
        else:
            bid_price = bid_price_1 + 1
        
        if ask_amount_1 < 5:
            ask_price = ask_price_2
        else:
            ask_price = ask_price_1 - 1

        return bid_price, ask_price
    

    def get_best_bid(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        #best_bid = max(market_bids)
        best_bid, best_bid_amount = list(market_bids.items())[0]

        return best_bid, best_bid_amount

    def get_best_ask(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        best_ask, best_ask_amount = list(market_asks.items())[0]

        return best_ask, best_ask_amount
    
    def get_mid_price(self, product, state : TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = self.DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2   

    def get_last_price(self, symbol, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        recent_trades = []
        if symbol in own_trades:
            recent_trades.extend(own_trades[symbol])
        if symbol in market_trades:
            recent_trades.extend(market_trades[symbol])
        recent_trades.sort(key=lambda trade: trade.timestamp)
        last_trade = recent_trades[-1]
        return last_trade.price
    
    
    def update_prices_history(self, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        for symbol in self.PRODUCTS:
            recent_trades = []
            if symbol in own_trades:
                recent_trades.extend(own_trades[symbol])
            if symbol in market_trades:
                recent_trades.extend(market_trades[symbol])

            recent_trades.sort(key=lambda trade: trade.timestamp)

            for trade in recent_trades:
                self.prices_history[symbol].append(trade.price)

            while len(self.prices_history[symbol]) > self.window_size:
                self.prices_history[symbol].pop(0)

    def update_mid_prices_history(self, state):
            for symbol in self.PRODUCTS:
                mid_price = self.get_mid_price(symbol, state)

                self.mid_prices_history[symbol].append(mid_price)

                while len(self.mid_prices_history[symbol]) > self.window_size:
                    self.mid_prices_history[symbol].pop(0)

    def update_diff_history(self, p_history):
        for symbol in self.PRODUCTS:
            if len(p_history[symbol]) >=2:
                diff = p_history[symbol][-1] - p_history[symbol][-2]
                
                self.diff_history[symbol].append(diff)

            while len(self.diff_history[symbol]) > 10:
                    self.diff_history[symbol].pop(0)


    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in self.PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]

        #print(self.ema_prices)

    def calculate_sma(self, product, window_size):
        sma = None
        prices = pd.Series(self.mid_prices_history[product])
        if len(prices) >= window_size:
            window_sum = prices.iloc[-window_size:].sum()
            sma = window_sum / window_size
        return sma
    
    def calculate_ema(self, product, window_size):
        ema = None
        prices = pd.Series(self.mid_prices_history[product])
        if len(prices) >= window_size:
            ema = prices.ewm(span=window_size, adjust=False).mean().iloc[-1]
        return ema
    

    def calculate_vwap(self, symbol, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        vwap = None
        recent_trades = []
        prices = []
        volumes = []
        if symbol in own_trades:
            recent_trades.extend(own_trades[symbol])
        if symbol in market_trades:
            recent_trades.extend(market_trades[symbol])

        recent_trades.sort(key=lambda trade: trade.timestamp)

        for trade in recent_trades:
            prices.append(trade.price)
            volumes.append(trade.quantity)

        data = pd.DataFrame({'prices': prices, 'volumes': volumes})
        vwap = (data['prices'] * data['volumes']).sum() / data['volumes'].sum()
        return vwap

    def calculate_standard_deviation(self, values: List[float]) -> float:
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]

        variance = sum(squared_diffs) / len(values)

        std_dev = math.sqrt(variance)

        return std_dev

    def calculate_order_book_imbalance(self, symbol, state: TradingState):
        if symbol not in state.order_depths:
            return None
        order_book = state.order_depths[symbol]
        bid_volume = sum(order_book.buy_orders.values())
        ask_volume = sum(order_book.sell_orders.values())

        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
            return imbalance
        else:
            print(total_volume)
            return 0
    
    def amethysts_strategy(self, state : TradingState) -> List[Order]:
        """
        Buying and Selling based on last trade price vs mean price (ceiling floor version)
        """
        orders = []
        position_amethysts = self.get_position('AMETHYSTS', state)

        bid_volume = self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        ask_volume = - self.POSITION_LIMITS['AMETHYSTS'] - position_amethysts
        last_price = self.get_last_price('AMETHYSTS', state.own_trades, state.market_trades)

        ema = self.ema_prices['AMETHYSTS']
        
        spread = 1
        open_spread = 3
        position_limit = 20
        position_spread = 16
        current_position = state.position.get("AMETHYSTS",0)
        best_ask = 0
        best_bid = 0
                
        order_depth_ame: OrderDepth = state.order_depths["AMETHYSTS"]
                
        # Check for anyone willing to sell for lower than 10 000 - 1
        if len(order_depth_ame.sell_orders) > 0:
            best_ask = min(order_depth_ame.sell_orders.keys())

            if best_ask <= 10000-spread:
                best_ask_volume = order_depth_ame.sell_orders[best_ask]
            else:
                best_ask_volume = 0
        else:
            best_ask_volume = 0

        # Check for buyers above 10 000 + 1
        if len(order_depth_ame.buy_orders) > 0:
            best_bid = max(order_depth_ame.buy_orders.keys())

            if best_bid >= 10000+spread:
                best_bid_volume = order_depth_ame.buy_orders[best_bid]
            else:
                best_bid_volume = 0 
        else:
            best_bid_volume = 0

        if current_position - best_ask_volume > position_limit:
            best_ask_volume = current_position - position_limit
            open_ask_volume = 0
        else:
            open_ask_volume = current_position - position_spread - best_ask_volume

        if current_position - best_bid_volume < -position_limit:
            best_bid_volume = current_position + position_limit
            open_bid_volume = 0
        else:
            open_bid_volume = current_position + position_spread - best_bid_volume

        if -open_ask_volume < 0:
            open_ask_volume = 0         
        if open_bid_volume < 0:
            open_bid_volume = 0

        if best_ask == 10000-open_spread and -best_ask_volume > 0:
            orders.append(Order("AMETHYSTS", 10000-open_spread, -best_ask_volume-open_ask_volume))
        else:
            if -best_ask_volume > 0:
                orders.append(Order("AMETHYSTS", best_ask, -best_ask_volume))
            if -open_ask_volume > 0:
                orders.append(Order("AMETHYSTS", 10000-open_spread, -open_ask_volume))

        if best_bid == 10000+open_spread and best_bid_volume > 0:
            orders.append(Order("AMETHYSTS", 10000+open_spread, -best_bid_volume-open_bid_volume))
        else:
            if best_bid_volume > 0:
                orders.append(Order("AMETHYSTS", best_bid, -best_bid_volume))
            if open_bid_volume > 0:
                orders.append(Order("AMETHYSTS", 10000+open_spread, -open_bid_volume))
    
        print(orders, last_price)
        
        return orders


    def starfruit_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of starfruit.
        """

        orders = []
    
        position_starfruit = self.get_position('STARFRUIT', state)

        bid_volume = self.POSITION_LIMITS['STARFRUIT'] - position_starfruit
        ask_volume = - self.POSITION_LIMITS['STARFRUIT'] - position_starfruit

        best_bid = self.get_best_bid('STARFRUIT', state)
        best_ask = self.get_best_ask('STARFRUIT', state)
        mid_price = self.get_mid_price('STARFRUIT', state)
        spread = (best_ask - best_bid) / 2
        last_price = self.get_last_price('STARFRUIT', state.own_trades, state.market_trades)


        if len(self.diff_history['STARFRUIT']) >= 6:
            AR_L1 = self.diff_history['STARFRUIT'][-1]
            AR_L2 = self.diff_history['STARFRUIT'][-2]
            AR_L3 = self.diff_history['STARFRUIT'][-3]
            AR_L4 = self.diff_history['STARFRUIT'][-4]
            AR_L5 = self.diff_history['STARFRUIT'][-5]
            AR_L6 = self.diff_history['STARFRUIT'][-6]
        
        if len(self.forecasted_diff_history['STARFRUIT']) > 0:
            forecasted_error = self.forecasted_diff_history['STARFRUIT'][-1] - self.diff_history['STARFRUIT'][-1]
            self.errors_history['STARFRUIT'].append(forecasted_error)

        if len(self.errors_history['STARFRUIT']) < 2:
            #use this!
            #self.errors_history['STARFRUIT'].extend([1.682936, -2.797327, -0.480615])
            
            self.errors_history['STARFRUIT'].extend([-4.218699, -0.945694, 0.298280])

            #self.errors_history['STARFRUIT'].extend([-3.258368, -3.353484, -3.593285])
      
        else:
            MA_L1 = self.errors_history['STARFRUIT'][-1]
            MA_L2 = self.errors_history['STARFRUIT'][-2]
            MA_L3 = self.errors_history['STARFRUIT'][-3]
        
        #use this!
        #forecasted_diff = (AR_L1 * -1.1102) + (AR_L2 * -0.7276) + (AR_L3 * -0.0854) + (AR_L4 * -0.0674)
        #+ (AR_L5 * -0.0437) + (AR_L6 * -0.0176)+ (MA_L1 *  0.4021) + (MA_L2 * -0.0587) + (MA_L3 * -0.4357)

        #new data
        forecasted_diff = (AR_L1 * -1.4799) + (AR_L2 * -0.8168) + (AR_L3 * -0.0868) + (AR_L4 * -0.0693)
        + (AR_L5 * -0.0492) + (AR_L6 * -0.0221)+ (MA_L1 *  0.7712) + (MA_L2 * -0.2324) + (MA_L3 * -0.4996)

        #price instead of mid price
        #forecasted_diff = (AR_L1 * -0.5038) + (AR_L2 * -0.4052) + (AR_L3 * -0.0110) + (AR_L4 * -0.0272)
        #+ (AR_L5 * -0.0209) + (AR_L6 * -0.0107)+ (MA_L1 *  -0.2193) + (MA_L2 * 0.0157) + (MA_L3 * -0.2858)

        self.forecasted_diff_history['STARFRUIT'].append(forecasted_diff)

        forecasted_price = mid_price + forecasted_diff  

        #play with diff comb
        if forecasted_price > best_bid+2:
            orders.append(Order('STARFRUIT', math.floor(best_bid+1), bid_volume))
            #orders.append(Order('STARFRUIT', math.floor(best_bid+2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+spread/2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.floor(forecasted_price+spread/3), int(math.ceil(ask_volume/2))))
        elif forecasted_price < best_ask-2:
            orders.append(Order('STARFRUIT', math.floor(best_ask-1), ask_volume))
            #orders.append(Order('STARFRUIT', math.floor(best_ask-2), int(math.floor(ask_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-spread/2), int(math.floor(bid_volume/2))))
            orders.append(Order('STARFRUIT', math.ceil(forecasted_price-spread/3), int(math.ceil(bid_volume/2))))
        

        return orders
    

    def orchids_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of orchids.
        """

        orders = []
    
        position_orchids = self.get_position('ORCHIDS', state)
        bid_price = self.get_best_bid('ORCHIDS', state)
        ask_price = self.get_best_ask('ORCHIDS', state)
        mid_price = self.get_mid_price('ORCHIDS', state)

        observations = state.observations
        conversion_observations = observations.conversionObservations
        orchid_observations = conversion_observations['ORCHIDS']

        bid_price_south = orchid_observations.bidPrice
        ask_price_south = orchid_observations.askPrice
        transport_fees = orchid_observations.transportFees
        export_tariff = orchid_observations.exportTariff
        import_tariff = orchid_observations.importTariff
        sunlight = orchid_observations.sunlight
        humidity = orchid_observations.humidity

        self.orchids_df = self.orchids_df.append(
            {'DAY': 2, 'timestamp': state.timestamp, 
            'ORCHIDS': mid_price, 'TRANSPORT_FEES': transport_fees, 
            'EXPORT_TARIFF': export_tariff, 
            'IMPORT_TARIFF': import_tariff,
             'SUNLIGHT': sunlight,
             'HUMIDITY': humidity,}, ignore_index=True)

        forecast = calculate_price(self.orchids_df)

        if forecasted_price > best_bid+1:
            orders.append(Order('ORCHIDS', math.floor(best_bid+1), bid_volume))
        elif forecasted_price < best_ask-1:
            orders.append(Order('ORCHIDS', math.floor(best_ask-1), ask_volume))
  
        return orders
    
    def orchids_strategy2(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of orchids.
        """
        orders = []
    
        position_orchids = self.get_position('ORCHIDS', state)

        bid_volume = self.POSITION_LIMITS['ORCHIDS'] - position_orchids
        ask_volume = - self.POSITION_LIMITS['ORCHIDS'] - position_orchids

        best_bid, best_bid_amount = self.get_best_bid('ORCHIDS', state)
        best_ask, best_ask_amount  = self.get_best_ask('ORCHIDS', state)

        mid_price = self.get_mid_price('ORCHIDS', state)

        spread = (best_ask - best_bid) / 2

        observations = state.observations
        conversion_observations = observations.conversionObservations
        orchid_observations = conversion_observations['ORCHIDS']

        bid_price_south = orchid_observations.bidPrice
        ask_price_south = orchid_observations.askPrice
        transport_fees = orchid_observations.transportFees
        export_tariff = orchid_observations.exportTariff
        import_tariff = orchid_observations.importTariff
        sunlight = orchid_observations.sunlight
        humidity = orchid_observations.humidity

        buy_price_south = ask_price_south + transport_fees + import_tariff
        sell_price_south = bid_price_south - transport_fees - export_tariff

        expected_profit_buying = 0
        expected_profit_selling = 0

        if position_orchids != 0:
            conversion = -position_orchids
        else:
            conversion = 0

        if best_ask < sell_price_south:
            #orders.append(Order('ORCHIDS', math.floor(best_ask), bid_volume))
            expected_profit_buying = sell_price_south - best_ask
        
        if best_bid > buy_price_south:
            #orders.append(Order('ORCHIDS', math.floor(best_bid), ask_volume))
            expected_profit_selling = best_bid - buy_price_south

        if expected_profit_buying > 0 and expected_profit_buying > expected_profit_selling:
            orders.append(Order('ORCHIDS', math.floor(best_ask), bid_volume))

        if expected_profit_selling > 0 and expected_profit_selling > expected_profit_buying:
            orders.append(Order('ORCHIDS', math.floor(best_bid), ask_volume))

        return orders, conversion

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : []}

        self.update_ema_prices(state)

        # PRICE HISTORY
        self.update_prices_history(state.own_trades, state.market_trades)
        self.update_mid_prices_history(state)
        #self.update_diff_history(self.mid_prices_history)
        self.update_diff_history(self.mid_prices_history)
        #print(self.prices_history)

        
        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp-100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.qt_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.current_pnl[product] -= trade.quantity * trade.price
                else:
                    self.current_pnl[product] += trade.quantity * trade.price
        
                final_pnl = 0
        for product in state.order_depths.keys():
            product_pnl = 0
            settled_pnl = 0
            best_sell = min(state.order_depths[product].sell_orders.keys())
            best_buy = max(state.order_depths[product].buy_orders.keys())
            mid_price = (best_sell + best_buy) / 2

            if self.get_position(product, state) < 0:
                settled_pnl += self.get_position(product, state) * mid_price
            else:
                settled_pnl += self.get_position(product, state) * mid_price
            product_pnl = settled_pnl + self.current_pnl[product]
            self.pnl_tracker[product].append(product_pnl)
            final_pnl += settled_pnl + self.current_pnl[product]
            print(f'\nFor product {product}, Pnl: {settled_pnl + self.current_pnl[product]}, Qty. Traded: {self.qt_traded[product]}')
        print(f'\nFinal Day Expected Pnl: {round(final_pnl,2)}')

        """

        

        for product in self.pnl_tracker.keys():
            while len(self.pnl_tracker[product]) > 10:
                self.pnl_tracker[product].pop(0)
            while len(self.forecasted_diff_history[product]) > 10:
                self.forecasted_diff_history[product].pop(0)
            while len(self.errors_history[product]) > 10:
                self.errors_history[product].pop(0)
        """   


        # AMETHYSTS STRATEGY
        """
        try:
            result['AMETHYSTS'] = self.amethysts_strategy(state)
        except Exception as e:
            print("Error in AMETHYSTS strategy")
            print(e)
        
        
        # STARFRUIT STRATEGY
        try:
            result['STARFRUIT'] = self.starfruit_strategy(state)
        except Exception as e:
            print("Error in STARFRUIT strategy")
            print(e)
        """

        # ORCHIDS STRATEGY
        try:
            result['ORCHIDS'] = self.orchids_strategy(state)
        except Exception as e:
            print("Error in ORCHIDS strategy")
            print(e)
                
        traderData = "SAMPLE" 
        
		# Sample conversion request. Check more details below. 
        conversions = 0
        return result, conversions, traderData

def calculate_la_stat_ewm(
    df: pd.DataFrame,
    col: str,
    window: int,
    span: float,
    prior_cc: float = 6,
    periods = 1,
    # max_periods = 10000,
) -> pd.DataFrame:
    hdf = df.copy()
    # compute diff from previous time
    hdf[col] = hdf[col].diff(periods=periods).fillna(0)

    # within season values
    day_ewm = hdf.groupby("DAY")[col].transform(lambda x: x.rolling(window=int(window), min_periods=1).apply(lambda y: custom_ewm(y, span), raw=True))
    day_ewm = (
        day_ewm.groupby([hdf["DAY"], hdf["timestamp"]])
        .last()
        .groupby("DAY")
        .shift()
        .reset_index(drop=False)
    )

    # cross season values
    ewm = hdf[col].transform(lambda x: x.rolling(window=int(window), min_periods=1).apply(lambda y: custom_ewm(y, span), raw=True))
    ewm = ewm.shift().ffill().reset_index(drop=False)

    day_cc = day_ewm.groupby("DAY").cumcount()

    # use cross season as prior
    ewm[col] = (
        (day_ewm[col] * day_cc + ewm[col] * prior_cc) / (day_cc + prior_cc)
    ).fillna(ewm[col])

    ewm.columns = ["Index", f"EWM_AVG_{col}"]

    ewm = ewm[[f"EWM_AVG_{col}"]]
    ewm = ewm.ffill().bfill()
    return ewm

def calculate_orchid_price(df):
    PARAMS = [ (669.92646802, 1977.98371139 , 926.31695217),  (493.57537655, 4405.02947201, 537.73672044),   (88.27088439,  567.85520167,  533.24102539)]
    param_groups = {
        "SUNLIGHT": PARAMS[0],
        "HUMIDITY": PARAMS[1],
        "ORCHIDS": PARAMS[2]
    }
    ewm_period = []
    for var in ["SUNLIGHT", "HUMIDITY", "ORCHIDS"]:
        window, span, prior_cc = param_groups[var]
        ewm_stat = calculate_la_stat_ewm(df, var, int(window), span, prior_cc)
        ewm_period.append(ewm_stat)
    ewm = pd.concat(ewm_period, axis=1)
    
    COEFS = [0.03925727, 0.00445401, 0.00366386]
    INTERCEPT = -0.018970904005940342
    ewm = ewm.iloc[-1]
    prev_price = df["ORCHIDS"].iloc[-1]
    all_neg = (ewm < 0).all() *  0.07510591
    return np.dot(ewm, COEFS) + all_neg + INTERCEPT + prev_price


def load_data():
    # Load the ORCHID_DATA string into a pandas DataFrame
    data = [line.split(";") for line in ORCHID_DATA.split("\n") if line]
    # timestamp;ORCHIDS;TRANSPORT_FEES;EXPORT_TARIFF;IMPORT_TARIFF;SUNLIGHT;HUMIDITY;DAY
    df = pd.DataFrame(data, columns=["timestamp", "ORCHIDS", "TRANSPORT_FEES", "EXPORT_TARIFF", "IMPORT_TARIFF", "SUNLIGHT", "HUMIDITY", "DAY"],
                      dtype=float)
    return df

def exp_weights(window, span):
    return np.exp(np.linspace(0, -window, window) / span)

# Custom function to apply with rolling
def custom_ewm(series, span):
    weights = exp_weights(len(series), span)
    return np.average(series, weights=weights)


ORCHID_DATA = """909800;1080.25;1.0;9.5;-5.0;3050.2056;71.1874;1
909900;1080.0;1.0;9.5;-5.0;3048.31;71.17869;1
910000;1077.25;1.0;9.5;-5.0;3046.4175;71.16997;1
910100;1077.5;1.0;9.5;-5.0;3044.5276;71.16124;1
910200;1076.25;1.0;9.5;-5.0;3042.6406;71.152504;1
910300;1075.75;1.0;9.5;-5.0;3040.7566;71.14376;1
910400;1075.75;1.0;9.5;-5.0;3038.8752;71.13502;1
910500;1076.25;1.0;9.5;-5.0;3036.9968;71.12627;1
910600;1075.75;1.0;9.5;-5.0;3035.121;71.11751;1
910700;1076.75;1.0;9.5;-5.0;3033.2483;71.10874;1
910800;1076.75;1.0;9.5;-5.0;3031.3784;71.099976;1
910900;1077.25;1.0;9.5;-5.0;3029.5112;71.0912;1
911000;1077.75;1.0;9.5;-5.0;3027.647;71.08243;1
911100;1076.75;1.0;9.5;-5.0;3025.7854;71.07364;1
911200;1076.75;1.0;9.5;-5.0;3023.9268;71.06485;1
911300;1075.75;1.0;9.5;-5.0;3022.0708;71.05606;1
911400;1076.75;1.0;9.5;-5.0;3020.2178;71.04726;1
911500;1076.25;1.0;9.5;-5.0;3018.3674;71.03845;1
911600;1076.5;1.0;9.5;-5.0;3016.52;71.02965;1
911700;1079.25;1.0;9.5;-5.0;3014.6755;71.02083;1
911800;1079.25;1.0;9.5;-5.0;3012.8337;71.01201;1
911900;1079.5;1.0;9.5;-5.0;3010.9949;71.00319;1
912000;1081.75;1.0;9.5;-5.0;3009.1587;70.99436;1
912100;1081.5;1.0;9.5;-5.0;3007.3252;70.98553;1
912200;1078.75;1.0;9.5;-5.0;3005.4946;70.97669;1
912300;1078.25;1.0;9.5;-5.0;3003.667;70.96785;1
912400;1078.5;1.0;9.5;-5.0;3001.842;70.95901;1
912500;1078.5;1.0;9.5;-5.0;3000.0198;70.95016;1
912600;1078.5;1.0;9.5;-5.0;2998.2004;70.9413;1
912700;1078.5;1.0;9.5;-5.0;2996.384;70.93244;1
912800;1078.25;1.0;9.5;-5.0;2994.57;70.923584;1
912900;1078.75;1.0;9.5;-5.0;2992.7593;70.91472;1
913000;1077.25;1.0;9.5;-5.0;2990.951;70.905846;1
913100;1077.75;1.0;9.5;-5.0;2989.1458;70.89697;1
913200;1078.0;1.0;9.5;-5.0;2987.343;70.88809;1
913300;1079.25;1.0;9.5;-5.0;2985.5432;70.87922;1
913400;1078.75;1.0;9.5;-5.0;2983.7463;70.87033;1
913500;1077.25;1.0;9.5;-5.0;2981.952;70.86144;1
913600;1078.25;1.0;9.5;-5.0;2980.1606;70.852554;1
913700;1078.25;1.0;9.5;-5.0;2978.3718;70.84366;1
913800;1079.75;1.0;9.5;-5.0;2976.586;70.83476;1
913900;1082.75;1.0;9.5;-5.0;2974.803;70.82586;1
914000;1083.25;1.0;9.5;-5.0;2973.0225;70.816956;1
914100;1086.25;1.0;9.5;-5.0;2971.245;70.80805;1
914200;1084.25;1.0;9.5;-5.0;2969.4702;70.79914;1
914300;1081.75;1.0;9.5;-5.0;2967.6982;70.79023;1
914400;1079.75;1.0;9.5;-5.0;2965.929;70.78131;1
914500;1080.25;1.0;9.5;-5.0;2964.1626;70.77239;1
914600;1079.75;1.0;9.5;-5.0;2962.3987;70.76347;1
914700;1080.75;1.0;9.5;-5.0;2960.638;70.754555;1
914800;1080.5;1.0;9.5;-5.0;2958.8796;70.74563;1
914900;1076.75;1.0;9.5;-5.0;2957.1243;70.7367;1
915000;1075.25;1.0;9.5;-5.0;2955.3716;70.72777;1
915100;1075.75;1.0;9.5;-5.0;2953.6216;70.718834;1
915200;1076.75;1.0;9.5;-5.0;2951.8745;70.7099;1
915300;1076.5;1.0;9.5;-5.0;2950.1301;70.700966;1
915400;1077.75;1.0;9.5;-5.0;2948.3884;70.692024;1
915500;1077.75;1.0;9.5;-5.0;2946.6497;70.68308;1
915600;1078.25;1.0;9.5;-5.0;2944.9136;70.67414;1
915700;1078.25;1.0;9.5;-5.0;2943.1802;70.6652;1
915800;1080.25;1.0;9.5;-5.0;2941.4495;70.65625;1
915900;1080.25;1.0;9.5;-5.0;2939.7217;70.6473;1
916000;1081.25;1.0;9.5;-5.0;2937.9966;70.63835;1
916100;1081.25;1.0;9.5;-5.0;2936.2742;70.6294;1
916200;1081.25;1.0;9.5;-5.0;2934.5544;70.62045;1
916300;1081.25;1.0;9.5;-5.0;2932.8376;70.611496;1
916400;1081.25;1.0;9.5;-5.0;2931.1233;70.60255;1
916500;1081.5;1.0;9.5;-5.0;2929.4119;70.59359;1
916600;1082.75;1.0;9.5;-5.0;2927.7031;70.58463;1
916700;1084.25;1.0;9.5;-5.0;2925.9973;70.57567;1
916800;1085.75;1.0;9.5;-5.0;2924.294;70.56671;1
916900;1086.0;1.0;9.5;-5.0;2922.5935;70.557755;1
917000;1086.0;1.0;9.5;-5.0;2920.8958;70.54879;1
917100;1084.75;1.0;9.5;-5.0;2919.2007;70.53983;1
917200;1086.25;1.0;8.5;-5.0;2917.5083;70.53087;1
917300;1088.75;1.0;8.5;-5.0;2915.8188;70.521904;1
917400;1088.75;1.0;8.5;-5.0;2914.1318;70.51294;1
917500;1087.75;1.0;8.5;-5.0;2912.4478;70.503975;1
917600;1085.75;1.0;8.5;-5.0;2910.7664;70.49501;1
917700;1084.75;1.0;8.5;-5.0;2909.0874;70.486046;1
917800;1083.75;1.0;8.5;-5.0;2907.4116;70.47708;1
917900;1085.75;1.0;8.5;-5.0;2905.7383;70.46812;1
918000;1086.75;1.0;8.5;-5.0;2904.0676;70.45915;1
918100;1086.25;1.0;8.5;-5.0;2902.3997;70.45019;1
918200;1087.75;1.0;8.5;-5.0;2900.7346;70.44122;1
918300;1087.75;1.0;8.5;-5.0;2899.0723;70.43226;1
918400;1086.25;1.0;8.5;-5.0;2897.4124;70.423294;1
918500;1086.25;1.0;8.5;-5.0;2895.7554;70.41433;1
918600;1084.75;1.0;8.5;-5.0;2894.101;70.405365;1
918700;1084.25;1.0;8.5;-5.0;2892.4495;70.3964;1
918800;1085.25;1.0;8.5;-5.0;2890.8005;70.387436;1
918900;1084.75;1.0;8.5;-5.0;2889.1543;70.37848;1
919000;1082.25;1.0;8.5;-5.0;2887.5107;70.369514;1
919100;1082.5;1.0;8.5;-5.0;2885.8699;70.36056;1
919200;1083.25;1.0;8.5;-5.0;2884.2317;70.35159;1
919300;1085.25;1.0;8.5;-5.5;2882.5962;70.342636;1
919400;1083.25;1.0;8.5;-5.5;2880.9634;70.33368;1
919500;1087.75;1.0;8.5;-5.5;2879.3333;70.32472;1
919600;1089.25;1.0;8.5;-5.5;2877.706;70.315765;1
919700;1088.25;1.0;8.5;-5.5;2876.0813;70.30681;1
919800;1085.75;1.0;8.5;-5.5;2874.4592;70.29786;1
919900;1086.75;1.0;8.5;-5.5;2872.8398;70.2889;1
920000;1088.25;1.0;8.5;-5.5;2871.2231;70.27995;1
920100;1088.75;1.0;8.5;-5.5;2869.6091;70.271;1
920200;1089.25;1.0;8.5;-5.5;2867.9978;70.262054;1
920300;1090.75;1.0;8.5;-5.5;2866.3894;70.25311;1
920400;1090.75;1.0;8.5;-5.5;2864.7834;70.24416;1
920500;1089.75;1.0;8.5;-5.5;2863.18;70.23522;1
920600;1088.75;1.0;8.5;-5.5;2861.5793;70.22628;1
920700;1091.25;1.0;8.5;-5.5;2859.9814;70.21735;1
920800;1089.75;1.0;8.5;-5.5;2858.3862;70.208405;1
920900;1087.75;1.0;8.5;-5.5;2856.7937;70.19947;1
921000;1088.75;1.0;8.5;-5.5;2855.2036;70.19054;1
921100;1090.25;1.0;8.5;-5.5;2853.6165;70.18161;1
921200;1092.25;1.0;8.5;-5.5;2852.0317;70.17268;1
921300;1089.25;1.0;8.5;-5.5;2850.4497;70.16376;1
921400;1090.25;1.0;8.5;-5.5;2848.8706;70.15483;1
921500;1090.25;1.0;8.5;-5.5;2847.294;70.14591;1
921600;1088.75;1.0;8.5;-5.5;2845.72;70.13699;1
921700;1088.5;1.0;8.5;-5.5;2844.1484;70.12808;1
921800;1086.75;1.0;8.5;-5.5;2842.5798;70.11917;1
921900;1083.25;1.0;8.5;-5.5;2841.014;70.11026;1
922000;1082.25;1.0;8.5;-5.5;2839.4504;70.10135;1
922100;1084.75;1.0;8.5;-5.5;2837.8896;70.092445;1
922200;1085.25;1.0;8.5;-5.5;2836.3315;70.08355;1
922300;1085.25;1.0;8.5;-5.5;2834.7761;70.07465;1
922400;1084.25;1.0;8.5;-5.5;2833.2234;70.06576;1
922500;1083.25;1.0;8.5;-5.5;2831.673;70.05687;1
922600;1083.5;1.0;8.5;-5.5;2830.1257;70.04798;1
922700;1081.25;1.0;8.5;-5.5;2828.5808;70.03909;1
922800;1078.25;1.0;8.5;-5.5;2827.0386;70.03021;1
922900;1076.25;1.0;8.5;-5.5;2825.499;70.02134;1
923000;1079.25;1.0;8.5;-5.5;2823.962;70.01247;1
923100;1079.0;1.0;8.5;-5.5;2822.4275;70.00359;1
923200;1076.25;1.0;8.5;-5.5;2820.8958;69.99473;1
923300;1077.75;1.0;8.5;-5.5;2819.3667;69.98587;1
923400;1076.75;1.0;8.5;-5.5;2817.8403;69.97701;1
923500;1076.75;1.0;8.5;-5.5;2816.3164;69.968155;1
923600;1077.25;1.0;8.5;-5.5;2814.7952;69.959305;1
923700;1077.25;1.0;8.5;-5.5;2813.2766;69.95046;1
923800;1077.5;1.0;8.5;-5.5;2811.7607;69.94162;1
923900;1077.75;1.0;8.5;-5.5;2810.2473;69.932785;1
924000;1076.25;1.0;8.5;-5.5;2808.7366;69.92395;1
924100;1078.25;1.0;8.5;-5.5;2807.2285;69.91512;1
924200;1077.25;1.0;8.5;-5.5;2805.723;69.9063;1
924300;1079.25;1.0;8.5;-5.5;2804.22;69.897484;1
924400;1079.25;1.0;8.5;-5.5;2802.7197;69.88867;1
924500;1078.75;1.0;8.5;-5.5;2801.222;69.87986;1
924600;1078.25;1.0;8.5;-5.5;2799.727;69.871056;1
924700;1076.25;1.0;8.5;-5.5;2798.2344;69.86226;1
924800;1075.25;1.0;8.5;-5.5;2796.7446;69.85346;1
924900;1076.75;1.0;8.5;-5.5;2795.2573;69.84467;1
925000;1076.75;1.0;8.5;-5.5;2793.7727;69.83589;1
925100;1075.25;1.0;8.5;-5.5;2792.2905;69.82711;1
925200;1073.25;1.0;8.5;-5.5;2790.811;69.81834;1
925300;1074.25;1.0;8.5;-5.5;2789.3342;69.80957;1
925400;1075.75;1.0;8.5;-5.5;2787.86;69.80081;1
925500;1077.25;1.0;8.5;-5.5;2786.3884;69.79205;1
925600;1075.75;1.0;8.5;-5.5;2784.9192;69.7833;1
925700;1075.5;1.0;8.5;-5.5;2783.4526;69.77456;1
925800;1076.25;1.0;8.5;-5.5;2781.9888;69.765816;1
925900;1076.25;1.0;8.5;-5.5;2780.5276;69.75708;1
926000;1075.25;1.0;8.5;-5.5;2779.0688;69.74835;1
926100;1077.25;1.0;8.5;-5.5;2777.6125;69.73963;1
926200;1077.25;1.0;8.5;-5.5;2776.159;69.73092;1
926300;1076.25;1.0;8.5;-5.5;2774.708;69.722206;1
926400;1075.25;1.0;8.5;-5.5;2773.2595;69.7135;1
926500;1075.75;1.0;8.5;-5.5;2771.8137;69.7048;1
926600;1076.25;1.0;8.5;-5.5;2770.3704;69.69611;1
926700;1076.0;1.0;8.5;-5.5;2768.9297;69.68743;1
926800;1077.75;1.0;8.5;-5.5;2767.4917;69.67876;1
926900;1078.75;1.0;8.5;-5.5;2766.0562;69.67008;1
927000;1078.75;1.0;8.5;-5.5;2764.623;69.661415;1
927100;1079.75;1.0;8.5;-5.5;2763.1926;69.65276;1
927200;1081.25;1.0;8.5;-5.5;2761.765;69.64411;1
927300;1080.25;1.0;8.5;-5.5;2760.3396;69.63547;1
927400;1080.75;1.0;8.5;-5.5;2758.9167;69.62683;1
927500;1082.25;1.0;8.5;-5.5;2757.4966;69.618195;1
927600;1081.75;1.0;8.5;-5.5;2756.079;69.60957;1
927700;1081.75;1.0;8.5;-5.5;2754.664;69.60096;1
927800;1081.75;1.0;8.5;-5.5;2753.2515;69.592354;1
927900;1080.25;1.0;8.5;-5.5;2751.8416;69.58375;1
928000;1080.75;1.0;8.5;-5.5;2750.4343;69.57516;1
928100;1079.25;1.0;8.5;-5.5;2749.0295;69.56657;1
928200;1078.75;1.0;8.5;-5.5;2747.6272;69.55799;1
928300;1076.75;1.0;8.5;-5.5;2746.2273;69.54942;1
928400;1074.25;1.0;8.5;-5.5;2744.8303;69.540855;1
928500;1073.25;1.0;8.5;-5.5;2743.4355;69.5323;1
928600;1074.25;1.0;8.5;-5.5;2742.0435;69.52376;1
928700;1074.5;1.0;8.5;-5.5;2740.654;69.51521;1
928800;1075.25;1.0;8.5;-5.5;2739.2668;69.50668;1
928900;1075.75;1.0;8.5;-5.5;2737.8826;69.49816;1
929000;1078.25;1.0;8.5;-5.5;2736.5005;69.48965;1
929100;1074.75;1.0;8.5;-5.5;2735.121;69.48114;1
929200;1074.5;1.0;8.5;-5.5;2733.7441;69.47264;1
929300;1075.25;1.0;8.5;-5.5;2732.3699;69.46415;1
929400;1074.25;1.0;8.5;-5.5;2730.998;69.455666;1
929500;1073.75;1.0;8.5;-5.5;2729.629;69.4472;1
929600;1074.75;1.0;8.5;-5.5;2728.262;69.43873;1
929700;1078.25;1.0;8.5;-5.5;2726.898;69.430275;1
929800;1079.75;1.0;8.5;-5.5;2725.5361;69.42183;1
929900;1078.75;1.0;8.5;-5.5;2724.177;69.41339;1
930000;1077.75;1.0;8.5;-5.5;2722.8203;69.40496;1
930100;1074.75;1.0;8.5;-5.5;2721.466;69.39654;1
930200;1074.75;1.0;8.5;-5.5;2720.1145;69.38813;1
930300;1075.75;1.0;8.5;-5.5;2718.7654;69.37972;1
930400;1073.25;1.0;8.5;-5.5;2717.419;69.37133;1
930500;1071.75;1.0;8.5;-5.5;2716.0747;69.362946;1
930600;1072.25;1.0;8.5;-5.5;2714.7332;69.354576;1
930700;1071.75;1.0;8.5;-5.5;2713.3943;69.34621;1
930800;1071.75;1.0;8.5;-5.5;2712.0576;69.33785;1
930900;1071.5;1.0;8.5;-5.5;2710.7236;69.329506;1
931000;1071.5;1.0;8.5;-5.5;2709.392;69.321175;1
931100;1071.25;1.0;8.5;-5.5;2708.063;69.31284;1
931200;1070.75;1.0;8.5;-5.5;2706.7366;69.30453;1
931300;1072.25;1.0;8.5;-5.5;2705.4124;69.29623;1
931400;1070.25;1.0;8.5;-5.5;2704.0908;69.287926;1
931500;1070.0;1.0;8.5;-5.5;2702.772;69.27964;1
931600;1071.75;1.0;8.5;-5.5;2701.4553;69.27136;1
931700;1070.25;1.0;8.5;-5.5;2700.1414;69.2631;1
931800;1069.75;1.0;8.5;-5.5;2698.8298;69.254845;1
931900;1068.75;1.0;8.5;-5.5;2697.5208;69.2466;1
932000;1070.25;1.0;8.5;-5.5;2696.214;69.23836;1
932100;1070.75;1.0;8.5;-5.5;2694.9102;69.23013;1
932200;1073.25;1.0;8.5;-5.5;2693.6084;69.221924;1
932300;1076.75;1.0;8.5;-5.5;2692.3093;69.21372;1
932400;1075.75;1.0;8.5;-5.5;2691.0127;69.20553;1
932500;1076.0;1.0;8.5;-5.5;2689.7185;69.19734;1
932600;1077.75;1.0;8.5;-5.5;2688.427;69.18917;1
932700;1079.25;1.0;8.5;-5.5;2687.1377;69.181015;1
932800;1078.25;1.0;8.5;-5.5;2685.851;69.17287;1
932900;1076.25;1.0;8.5;-5.5;2684.5667;69.16473;1
933000;1073.75;1.0;8.5;-5.5;2683.285;69.1566;1
933100;1074.0;1.0;8.5;-5.5;2682.0056;69.14848;1
933200;1072.25;1.0;8.5;-5.5;2680.7288;69.14038;1
933300;1073.75;1.0;8.5;-5.5;2679.4546;69.132286;1
933400;1073.75;1.0;8.5;-5.5;2678.1826;69.12421;1
933500;1072.25;1.0;8.5;-5.5;2676.913;69.116135;1
933600;1072.25;1.0;8.5;-5.5;2675.6462;69.10808;1
933700;1072.75;1.0;8.5;-5.5;2674.3816;69.10003;1
933800;1073.75;1.0;8.5;-5.5;2673.1196;69.091995;1
933900;1073.5;0.9;8.5;-5.5;2671.86;69.08398;1
934000;1071.75;0.9;8.5;-5.5;2670.603;69.075966;1
934100;1072.75;0.9;8.5;-5.5;2669.3484;69.06796;1
934200;1072.75;0.9;8.5;-5.5;2668.0962;69.05998;1
934300;1071.75;0.9;8.5;-5.5;2666.8464;69.052;1
934400;1073.25;0.9;8.5;-5.5;2665.599;69.044044;1
934500;1072.25;0.9;8.5;-5.5;2664.3542;69.036095;1
934600;1070.25;0.9;8.5;-5.5;2663.1118;69.02815;1
934700;1070.0;0.9;8.5;-5.5;2661.8718;69.020226;1
934800;1068.25;0.9;8.5;-5.5;2660.6343;69.012314;1
934900;1068.25;0.9;8.5;-5.5;2659.3992;69.00442;1
935000;1064.75;0.9;8.5;-5.5;2658.1665;68.99653;1
935100;1063.25;0.9;8.5;-5.5;2656.9363;68.988655;1
935200;1061.75;0.9;8.5;-5.5;2655.7085;68.98079;1
935300;1060.75;0.9;8.5;-5.5;2654.4832;68.972946;1
935400;1061.0;0.9;8.5;-5.5;2653.2603;68.96511;1
935500;1060.75;0.9;8.5;-5.5;2652.0398;68.95728;1
935600;1060.75;0.9;8.5;-5.5;2650.8218;68.94948;1
935700;1061.75;0.9;8.5;-5.5;2649.6062;68.94168;1
935800;1062.75;0.9;8.5;-5.5;2648.393;68.93389;1
935900;1064.25;0.9;8.5;-5.5;2647.1824;68.926125;1
936000;1064.5;0.9;8.5;-5.5;2645.9739;68.918365;1
936100;1064.5;0.9;8.5;-5.5;2644.768;68.91062;1
936200;1064.75;0.9;8.5;-5.5;2643.5647;68.90289;1
936300;1067.25;0.9;8.5;-5.5;2642.3635;68.89518;1
936400;1068.75;0.9;8.5;-5.5;2641.165;68.887474;1
936500;1071.25;0.9;8.5;-5.5;2639.9688;68.87979;1
936600;1069.25;0.9;8.5;-5.5;2638.775;68.872116;1
936700;1070.25;0.9;8.5;-5.5;2637.5835;68.864456;1
936800;1069.75;0.9;8.5;-5.5;2636.3945;68.85681;1
936900;1069.25;0.9;8.5;-5.5;2635.208;68.849174;1
937000;1070.25;0.9;8.5;-5.5;2634.024;68.84156;1
937100;1070.25;0.9;8.5;-5.5;2632.842;68.833954;1
937200;1067.75;0.9;8.5;-5.5;2631.6628;68.82636;1
937300;1068.0;0.9;8.5;-5.5;2630.4858;68.81879;1
937400;1069.25;0.9;8.5;-5.5;2629.3113;68.811226;1
937500;1068.25;0.9;8.5;-5.5;2628.1392;68.80368;1
937600;1066.25;0.9;8.5;-5.5;2626.9695;68.79615;1
937700;1067.25;0.9;8.5;-5.5;2625.802;68.788635;1
937800;1065.25;0.9;8.5;-5.5;2624.6372;68.78113;1
937900;1064.25;0.9;8.5;-5.5;2623.4746;68.77364;1
938000;1063.75;0.9;8.5;-5.5;2622.3145;68.766174;1
938100;1063.75;0.9;8.5;-5.5;2621.1567;68.75871;1
938200;1063.25;0.9;8.5;-5.5;2620.0015;68.751274;1
938300;1064.25;0.9;8.5;-5.5;2618.8484;68.74384;1
938400;1064.5;0.9;8.5;-5.5;2617.6978;68.736435;1
938500;1062.25;0.9;8.5;-5.5;2616.5496;68.729034;1
938600;1064.75;0.9;8.5;-5.5;2615.4038;68.72166;1
938700;1066.25;0.9;8.5;-5.5;2614.2603;68.71429;1
938800;1063.75;0.9;8.5;-5.5;2613.1191;68.70694;1
938900;1060.75;0.9;8.5;-5.5;2611.9805;68.69961;1
939000;1061.75;0.9;8.5;-5.5;2610.8442;68.69228;1
939100;1063.75;0.9;8.5;-5.5;2609.7102;68.68498;1
939200;1064.75;0.9;8.5;-5.5;2608.5789;68.6777;1
939300;1065.0;0.9;8.5;-5.5;2607.4497;68.670425;1
939400;1064.25;0.9;8.5;-5.5;2606.3228;68.66317;1
939500;1064.25;0.9;8.5;-5.5;2605.1982;68.65593;1
939600;1066.75;0.9;8.5;-5.5;2604.0762;68.648705;1
939700;1064.25;0.9;8.5;-5.5;2602.9565;68.6415;1
939800;1068.25;0.9;8.5;-5.5;2601.839;68.63431;1
939900;1069.75;0.9;8.5;-5.5;2600.724;68.62714;1
940000;1069.75;0.9;8.5;-5.5;2599.6116;68.61998;1
940100;1069.75;0.9;8.5;-5.5;2598.5015;68.61283;1
940200;1068.25;0.9;8.5;-5.5;2597.3936;68.60571;1
940300;1066.75;0.9;8.5;-5.5;2596.2878;68.5986;1
940400;1066.25;0.9;8.5;-5.5;2595.1848;68.59151;1
940500;1067.75;0.9;8.5;-5.5;2594.084;68.584435;1
940600;1067.25;0.9;8.5;-5.0;2592.9854;68.57738;1
940700;1068.25;0.9;8.5;-5.0;2591.8892;68.570335;1
940800;1064.75;0.9;8.5;-5.0;2590.7954;68.56331;1
940900;1065.25;0.9;8.5;-5.0;2589.704;68.556305;1
941000;1062.25;0.9;8.5;-5.0;2588.615;68.54932;1
941100;1063.25;0.9;8.5;-5.0;2587.528;68.54234;1
941200;1062.75;0.9;8.5;-5.0;2586.4438;68.535385;1
941300;1059.75;0.9;8.5;-5.0;2585.3618;68.52845;1
941400;1059.25;0.9;8.5;-5.0;2584.282;68.52153;1
941500;1056.75;0.9;8.5;-5.0;2583.2046;68.514626;1
941600;1057.25;0.9;8.5;-5.0;2582.1296;68.507744;1
941700;1057.25;0.9;8.5;-5.0;2581.057;68.50088;1
941800;1055.75;0.9;8.5;-5.0;2579.9863;68.494026;1
941900;1054.75;0.9;8.5;-5.0;2578.9185;68.4872;1
942000;1054.75;0.9;8.5;-5.0;2577.8525;68.48038;1
942100;1054.75;0.9;8.5;-5.0;2576.7893;68.47359;1
942200;1053.25;0.9;8.5;-5.0;2575.728;68.466805;1
942300;1052.25;0.9;8.5;-5.0;2574.6694;68.460045;1
942400;1048.25;0.9;8.5;-5.0;2573.613;68.45331;1
942500;1048.25;0.9;8.5;-5.0;2572.5588;68.44659;1
942600;1048.25;0.9;8.5;-5.0;2571.507;68.43988;1
942700;1048.0;0.9;8.5;-5.0;2570.4575;68.43319;1
942800;1047.25;0.9;8.5;-5.0;2569.4104;68.42652;1
942900;1046.25;0.9;8.5;-5.0;2568.3657;68.419876;1
943000;1046.75;0.9;8.5;-5.0;2567.323;68.413246;1
943100;1048.75;0.9;8.5;-5.0;2566.283;68.40663;1
943200;1047.75;0.9;8.5;-5.0;2565.2449;68.40004;1
943300;1048.75;0.9;8.5;-5.0;2564.2095;68.39347;1
943400;1049.75;0.9;8.5;-5.0;2563.176;68.38691;1
943500;1048.25;0.9;8.5;-5.0;2562.145;68.38038;1
943600;1048.25;0.9;8.5;-5.0;2561.1165;68.37386;1
943700;1050.75;0.9;8.5;-5.0;2560.09;68.36736;1
943800;1052.25;0.9;8.5;-5.0;2559.066;68.360886;1
943900;1051.75;0.9;8.5;-5.0;2558.0442;68.35442;1
944000;1052.0;0.9;8.5;-5.0;2557.0247;68.347984;1
944100;1050.25;0.9;8.5;-5.0;2556.0076;68.34157;1
944200;1050.0;0.9;8.5;-5.0;2554.9927;68.33517;1
944300;1050.0;0.9;8.5;-5.0;2553.98;68.32878;1
944400;1047.75;0.9;8.5;-5.0;2552.9697;68.322426;1
944500;1048.75;0.9;8.5;-5.0;2551.9617;68.31608;1
944600;1048.25;0.9;8.5;-5.0;2550.956;68.30976;1
944700;1047.75;0.9;8.5;-5.0;2549.9526;68.30346;1
944800;1049.25;0.9;8.5;-5.0;2548.9514;68.29718;1
944900;1048.75;0.9;8.5;-5.0;2547.9526;68.29092;1
945000;1045.25;0.9;8.5;-5.0;2546.956;68.284676;1
945100;1044.75;0.9;8.5;-5.0;2545.9617;68.27846;1
945200;1043.25;0.9;8.5;-5.0;2544.9697;68.272255;1
945300;1044.75;0.9;8.5;-5.0;2543.98;68.266075;1
945400;1044.75;0.9;8.5;-5.0;2542.9924;68.25991;1
945500;1048.25;0.9;8.5;-5.0;2542.007;68.25378;1
945600;1049.25;0.9;8.5;-5.0;2541.0242;68.24766;1
945700;1051.25;0.9;8.5;-5.0;2540.0437;68.24156;1
945800;1050.25;0.9;8.5;-5.0;2539.0652;68.23548;1
945900;1048.25;0.9;8.5;-5.0;2538.089;68.22942;1
946000;1047.25;0.9;8.5;-5.0;2537.1152;68.22339;1
946100;1046.25;0.9;8.5;-5.0;2536.1436;68.21738;1
946200;1048.25;0.9;8.5;-5.0;2535.1743;68.21138;1
946300;1048.5;0.9;8.5;-5.0;2534.2073;68.205414;1
946400;1046.75;0.9;8.5;-5.0;2533.2424;68.19946;1
946500;1046.75;0.9;8.5;-5.0;2532.2798;68.19353;1
946600;1046.75;0.9;8.5;-5.0;2531.3196;68.18762;1
946700;1048.25;0.9;8.5;-5.0;2530.3616;68.18173;1
946800;1049.25;0.9;8.5;-5.0;2529.4058;68.17587;1
946900;1049.25;0.9;8.5;-5.0;2528.4521;68.17003;1
947000;1049.0;0.9;8.5;-5.0;2527.5007;68.16421;1
947100;1050.75;0.9;8.5;-5.0;2526.5518;68.1584;1
947200;1053.75;0.9;8.5;-5.0;2525.605;68.152626;1
947300;1052.25;0.9;8.5;-5.0;2524.6604;68.14687;1
947400;1054.75;0.9;8.5;-5.0;2523.718;68.141136;1
947500;1057.25;0.9;8.5;-5.0;2522.778;68.13542;1
947600;1058.75;0.9;8.5;-5.0;2521.84;68.12973;1
947700;1060.25;0.9;8.5;-5.0;2520.9045;68.12406;1
947800;1063.25;0.9;8.5;-5.0;2519.9712;68.118416;1
947900;1064.75;0.9;8.5;-5.0;2519.04;68.11279;1
948000;1065.0;0.9;8.5;-5.0;2518.1113;68.10719;1
948100;1067.25;0.9;8.5;-5.0;2517.1846;68.101616;1
948200;1067.5;0.9;8.5;-5.0;2516.26;68.09606;1
948300;1065.75;0.9;8.5;-5.0;2515.338;68.09053;1
948400;1067.25;0.9;8.5;-5.0;2514.418;68.085014;1
948500;1065.25;0.9;8.5;-5.0;2513.5002;68.07953;1
948600;1067.25;0.9;8.5;-5.0;2512.5847;68.074066;1
948700;1068.75;0.9;8.5;-5.0;2511.6714;68.06862;1
948800;1070.75;0.9;8.5;-5.0;2510.7603;68.0632;1
948900;1072.25;0.9;8.5;-5.0;2509.8516;68.05781;1
949000;1072.5;0.9;8.5;-5.0;2508.9448;68.05244;1
949100;1072.75;0.9;8.5;-5.0;2508.0403;68.04709;1
949200;1072.75;0.9;8.5;-5.0;2507.1382;68.04176;1
949300;1072.25;0.9;8.5;-5.0;2506.2383;68.03645;1
949400;1073.75;0.9;8.5;-5.0;2505.3403;68.03118;1
949500;1072.25;0.9;8.5;-5.0;2504.4448;68.025925;1
949600;1073.25;0.9;8.5;-5.0;2503.5515;68.02069;1
949700;1074.25;0.9;8.5;-5.0;2502.6602;68.01548;1
949800;1073.25;0.9;8.5;-5.0;2501.7712;68.0103;1
949900;1079.75;0.9;8.5;-5.0;2500.8845;68.005135;1
950000;1080.0;0.9;8.5;-5.0;2500.0;68.0;1
950100;1081.25;0.9;8.5;-5.0;2499.1177;67.99489;1
950200;1081.75;0.9;8.5;-5.0;2498.2375;67.9898;1
950300;1082.75;0.9;8.5;-5.0;2497.3596;67.98473;1
950400;1081.75;0.9;8.5;-5.0;2496.484;67.97969;1
950500;1080.75;0.9;8.5;-5.0;2495.61;67.97468;1
950600;1081.25;0.9;8.5;-5.0;2494.7388;67.96969;1
950700;1081.75;0.9;8.5;-5.0;2493.8696;67.96472;1
950800;1081.75;0.9;8.5;-5.0;2493.0027;67.95978;1
950900;1083.75;0.9;8.5;-5.0;2492.138;67.95486;1
951000;1084.75;0.9;8.5;-5.0;2491.2751;67.94997;1
951100;1083.75;0.9;8.5;-5.0;2490.4148;67.9451;1
951200;1083.5;0.9;8.5;-5.0;2489.5564;67.940254;1
951300;1084.25;0.9;8.5;-5.0;2488.7004;67.93543;1
951400;1086.25;0.9;8.5;-5.0;2487.8464;67.93064;1
951500;1086.25;0.9;8.5;-5.0;2486.9949;67.92587;1
951600;1086.25;0.9;8.5;-5.0;2486.1453;67.92113;1
951700;1086.75;0.9;8.5;-5.0;2485.2979;67.91641;1
951800;1084.25;0.9;8.5;-5.0;2484.4526;67.91171;1
951900;1085.75;0.9;8.5;-5.0;2483.6096;67.90705;1
952000;1084.75;0.9;8.5;-5.0;2482.7688;67.902405;1
952100;1083.75;0.9;8.5;-5.0;2481.9302;67.89779;1
952200;1086.25;0.9;8.5;-5.0;2481.0935;67.893196;1
952300;1085.75;0.9;8.5;-5.0;2480.2593;67.88863;1
952400;1082.75;0.9;8.5;-5.0;2479.427;67.884094;1
952500;1083.75;0.9;8.5;-5.0;2478.597;67.87958;1
952600;1083.25;0.9;8.5;-5.0;2477.769;67.87509;1
952700;1080.75;0.9;8.5;-5.0;2476.9434;67.87063;1
952800;1083.25;0.9;8.5;-5.0;2476.1199;67.86619;1
952900;1085.25;0.9;8.5;-5.0;2475.2983;67.86178;1
953000;1084.25;0.9;8.5;-5.0;2474.4792;67.85739;1
953100;1083.25;0.9;8.5;-5.0;2473.662;67.853035;1
953200;1083.5;0.9;8.5;-5.0;2472.8472;67.84871;1
953300;1084.25;0.9;8.5;-5.0;2472.0344;67.8444;1
953400;1085.25;0.9;8.5;-5.0;2471.2236;67.84012;1
953500;1085.5;0.9;8.5;-5.0;2470.4153;67.83587;1
953600;1085.5;0.9;8.5;-5.0;2469.609;67.83164;1
953700;1085.75;0.9;8.5;-5.0;2468.8047;67.827446;1
953800;1086.0;0.9;8.5;-5.0;2468.0024;67.82327;1
953900;1088.75;0.9;8.5;-5.0;2467.2026;67.81913;1
954000;1087.75;0.9;8.5;-5.0;2466.4048;67.81501;1
954100;1088.75;0.9;8.5;-5.0;2465.6091;67.81092;1
954200;1089.75;0.9;8.5;-5.0;2464.8157;67.806854;1
954300;1090.25;0.9;8.5;-5.0;2464.0242;67.80282;1
954400;1092.75;0.9;8.5;-5.0;2463.2349;67.798805;1
954500;1091.75;0.9;8.5;-5.0;2462.4478;67.79482;1
954600;1091.75;0.9;8.5;-5.0;2461.6628;67.79087;1
954700;1091.75;0.9;8.5;-5.0;2460.88;67.78694;1
954800;1093.75;0.9;8.5;-5.0;2460.099;67.78304;1
954900;1091.25;0.9;8.5;-5.0;2459.3206;67.77917;1
955000;1092.75;0.9;8.5;-5.0;2458.544;67.77532;1
955100;1095.25;0.9;8.5;-5.0;2457.7695;67.77151;1
955200;1096.25;0.9;8.5;-5.0;2456.9973;67.767715;1
955300;1097.25;0.9;8.5;-5.0;2456.2273;67.763954;1
955400;1096.25;0.9;8.5;-5.0;2455.4592;67.76022;1
955500;1095.75;0.9;8.5;-5.0;2454.693;67.756516;1
955600;1092.75;0.9;8.5;-5.0;2453.9294;67.75284;1
955700;1093.25;0.9;8.5;-5.0;2453.1677;67.74919;1
955800;1092.75;0.9;8.5;-5.0;2452.4082;67.74557;1
955900;1093.75;0.9;8.5;-5.0;2451.6506;67.74198;1
956000;1092.25;0.9;8.5;-5.0;2450.8953;67.73841;1
956100;1094.75;0.9;8.5;-5.0;2450.142;67.73488;1
956200;1094.75;0.9;8.5;-5.0;2449.3909;67.73137;1
956300;1094.5;0.9;8.5;-5.0;2448.6418;67.7279;1
956400;1096.75;0.9;8.5;-5.0;2447.8948;67.72444;1
956500;1095.75;0.9;8.5;-5.0;2447.15;67.72102;1
956600;1095.75;0.9;8.5;-5.0;2446.407;67.71763;1
956700;1095.5;0.9;8.5;-5.0;2445.6665;67.71427;1
956800;1098.25;0.9;8.5;-5.0;2444.9277;67.71094;1
956900;1098.75;0.9;8.5;-5.0;2444.1914;67.70763;1
957000;1098.5;0.9;8.5;-5.0;2443.4568;67.70435;1
957100;1099.75;0.9;8.5;-5.0;2442.7246;67.70111;1
957200;1098.25;0.9;8.5;-5.0;2441.9944;67.69789;1
957300;1097.75;0.9;8.5;-5.0;2441.266;67.6947;1
957400;1098.0;0.9;8.5;-5.0;2440.54;67.69154;1
957500;1098.0;0.9;8.5;-5.0;2439.8162;67.688416;1
957600;1097.25;0.9;8.5;-5.0;2439.0942;67.68532;1
957700;1096.25;0.9;8.5;-5.0;2438.3745;67.68225;1
957800;1094.25;0.9;8.5;-5.0;2437.6567;67.67921;1
957900;1094.5;0.9;8.5;-5.0;2436.941;67.6762;1
958000;1093.75;0.9;8.5;-5.0;2436.2273;67.67322;1
958100;1093.5;0.9;8.5;-5.0;2435.5159;67.67027;1
958200;1092.75;0.9;8.5;-5.0;2434.8064;67.66735;1
958300;1092.25;0.9;8.5;-5.0;2434.0989;67.66446;1
958400;1090.25;0.9;8.5;-5.0;2433.3938;67.6616;1
958500;1091.75;0.9;8.5;-5.0;2432.6904;67.658775;1
958600;1093.25;0.9;8.5;-5.0;2431.9893;67.655975;1
958700;1091.25;0.9;8.5;-5.0;2431.29;67.653206;1
958800;1091.0;0.9;8.5;-5.0;2430.593;67.65047;1
958900;1088.75;0.9;8.5;-5.0;2429.898;67.64776;1
959000;1088.25;0.9;8.5;-5.0;2429.205;67.64508;1
959100;1086.75;0.9;8.5;-5.0;2428.5142;67.64244;1
959200;1086.5;0.9;8.5;-5.0;2427.8254;67.639824;1
959300;1087.25;0.9;8.5;-5.0;2427.1387;67.63724;1
959400;1089.25;0.9;8.5;-5.0;2426.4539;67.63469;1
959500;1088.75;0.9;8.5;-5.0;2425.7712;67.632164;1
959600;1085.75;0.9;8.5;-5.0;2425.0906;67.62968;1
959700;1084.25;0.9;8.5;-5.0;2424.4119;67.62721;1
959800;1084.25;0.9;8.5;-5.0;2423.7354;67.62479;1
959900;1084.5;0.9;8.5;-5.0;2423.0608;67.62239;1
960000;1084.5;0.9;8.5;-5.0;2422.3884;67.620026;1
960100;1082.75;0.9;8.5;-5.0;2421.718;67.61769;1
960200;1083.75;0.9;8.5;-5.0;2421.0496;67.61539;1
960300;1085.25;0.9;8.5;-5.0;2420.3833;67.61311;1
960400;1085.75;0.9;8.5;-5.0;2419.719;67.61088;1
960500;1085.25;0.9;8.5;-5.0;2419.0566;67.60867;1
960600;1081.75;0.9;8.5;-5.0;2418.3965;67.60649;1
960700;1080.75;0.9;8.5;-5.0;2417.7383;67.604355;1
960800;1078.75;0.9;8.5;-5.0;2417.082;67.60224;1
960900;1077.25;0.9;8.5;-5.0;2416.428;67.60016;1
961000;1078.75;0.9;8.5;-5.0;2415.7756;67.598114;1
961100;1078.25;0.9;8.5;-5.0;2415.1257;67.5961;1
961200;1078.25;0.9;8.5;-5.0;2414.4775;67.59412;1
961300;1080.75;0.9;8.5;-5.0;2413.8315;67.59216;1
961400;1079.75;0.9;8.5;-5.0;2413.1875;67.59025;1
961500;1081.25;0.9;8.5;-5.0;2412.5454;67.58836;1
961600;1082.25;0.9;8.5;-5.0;2411.9055;67.58651;1
961700;1082.25;0.9;8.5;-5.0;2411.2673;67.584694;1
961800;1083.25;0.9;8.5;-5.0;2410.6313;67.5829;1
961900;1085.75;0.9;9.5;-5.0;2409.9976;67.58115;1
962000;1086.0;0.9;9.5;-5.0;2409.3655;67.57943;1
962100;1087.25;0.9;9.5;-5.0;2408.7356;67.57774;1
962200;1088.25;0.9;9.5;-5.0;2408.1077;67.57609;1
962300;1088.5;0.9;9.5;-5.0;2407.4817;67.57446;1
962400;1088.5;0.9;9.5;-5.0;2406.8577;67.572876;1
962500;1085.75;0.9;9.5;-5.0;2406.2358;67.57132;1
962600;1084.25;0.9;9.5;-5.0;2405.616;67.56979;1
962700;1082.75;0.9;9.5;-5.0;2404.998;67.568306;1
962800;1081.75;0.9;9.5;-5.0;2404.382;67.56685;1
962900;1081.25;0.9;9.5;-5.0;2403.768;67.56543;1
963000;1081.75;0.9;9.5;-5.0;2403.1562;67.56404;1
963100;1082.75;0.9;9.5;-5.0;2402.5461;67.56269;1
963200;1082.75;0.9;9.5;-5.0;2401.9382;67.56137;1
963300;1084.25;0.9;9.5;-5.0;2401.3323;67.56008;1
963400;1084.25;0.9;9.5;-5.0;2400.7283;67.55883;1
963500;1082.25;0.9;9.5;-5.0;2400.1262;67.55761;1
963600;1082.0;0.9;9.5;-5.0;2399.5264;67.55643;1
963700;1080.75;0.9;9.5;-5.0;2398.9282;67.555275;1
963800;1080.25;0.9;9.5;-5.0;2398.3323;67.55416;1
963900;1078.75;0.9;9.5;-5.0;2397.738;67.55308;1
964000;1076.75;0.9;9.5;-5.0;2397.146;67.55203;1
964100;1078.25;0.9;9.5;-5.0;2396.556;67.55102;1
964200;1079.75;0.9;9.5;-5.0;2395.9678;67.55004;1
964300;1079.75;0.9;9.5;-5.0;2395.3816;67.549095;1
964400;1078.25;0.9;9.5;-5.0;2394.7974;67.54819;1
964500;1076.75;0.9;9.5;-5.0;2394.215;67.54732;1
964600;1077.0;0.9;9.5;-5.0;2393.6348;67.54648;1
964700;1078.25;0.9;9.5;-5.0;2393.0566;67.54567;1
964800;1080.25;0.9;9.5;-5.0;2392.4802;67.54491;1
964900;1078.75;0.9;9.5;-5.0;2391.906;67.54417;1
965000;1078.25;0.9;9.5;-5.0;2391.3335;67.54347;1
965100;1076.75;0.9;9.5;-5.0;2390.763;67.54281;1
965200;1077.25;0.9;9.5;-5.0;2390.1946;67.54218;1
965300;1076.75;0.9;9.5;-5.0;2389.628;67.54159;1
965400;1078.25;0.9;9.5;-5.0;2389.0635;67.54103;1
965500;1075.75;0.9;9.5;-5.0;2388.501;67.540504;1
965600;1074.75;0.9;9.5;-5.0;2387.9402;67.54002;1
965700;1074.75;0.9;9.5;-5.0;2387.3816;67.53957;1
965800;1075.0;0.9;9.5;-5.0;2386.8247;67.539154;1
965900;1074.25;0.9;9.5;-5.0;2386.27;67.53878;1
966000;1075.25;0.9;9.5;-5.0;2385.717;67.53844;1
966100;1073.75;0.9;9.5;-5.0;2385.166;67.53813;1
966200;1074.75;0.9;9.5;-5.0;2384.6172;67.53786;1
966300;1073.25;0.9;9.5;-5.0;2384.07;67.53762;1
966400;1073.25;0.9;9.5;-5.0;2383.525;67.53742;1
966500;1075.25;0.9;9.5;-5.0;2382.982;67.53726;1
966600;1077.25;0.9;9.5;-5.0;2382.4407;67.53714;1
966700;1075.75;0.9;9.5;-5.0;2381.9014;67.53705;1
966800;1075.75;0.9;9.5;-5.0;2381.364;67.536995;1
966900;1074.75;0.9;9.5;-5.0;2380.8286;67.53698;1
967000;1072.75;0.9;9.5;-5.0;2380.295;67.537;1
967100;1073.75;0.9;9.5;-5.0;2379.7634;67.537056;1
967200;1075.25;0.9;9.5;-5.0;2379.234;67.53715;1
967300;1075.25;0.9;9.5;-5.0;2378.706;67.537285;1
967400;1076.25;0.9;9.5;-5.0;2378.1804;67.53745;1
967500;1076.25;0.9;9.5;-5.0;2377.6565;67.53766;1
967600;1072.75;0.9;9.5;-5.0;2377.1345;67.537895;1
967700;1070.25;0.9;9.5;-5.0;2376.6145;67.53818;1
967800;1071.25;0.9;9.5;-5.0;2376.0964;67.53849;1
967900;1069.75;0.9;9.5;-5.0;2375.58;67.53885;1
968000;1069.25;0.9;9.5;-5.0;2375.066;67.53924;1
968100;1069.5;0.9;9.5;-5.0;2374.5535;67.539665;1
968200;1070.25;0.9;9.5;-5.0;2374.043;67.54013;1
968300;1070.5;0.9;9.5;-5.0;2373.5344;67.54064;1
968400;1073.25;0.9;9.5;-5.0;2373.0278;67.54118;1
968500;1075.25;0.9;9.5;-5.0;2372.5232;67.54176;1
968600;1073.75;0.9;9.5;-5.0;2372.0203;67.54237;1
968700;1074.75;0.9;9.5;-5.0;2371.5193;67.54303;1
968800;1071.75;0.9;9.5;-5.0;2371.0203;67.543724;1
968900;1070.75;0.9;9.5;-5.0;2370.5232;67.54446;1
969000;1071.25;0.9;9.5;-5.0;2370.0278;67.54523;1
969100;1071.5;0.9;9.5;-5.0;2369.5347;67.546036;1
969200;1071.25;0.9;9.5;-5.0;2369.0432;67.54688;1
969300;1072.25;0.9;9.5;-5.0;2368.5537;67.54777;1
969400;1072.75;0.9;9.5;-5.0;2368.066;67.54869;1
969500;1069.75;0.9;9.5;-5.0;2367.5803;67.54965;1
969600;1069.75;0.9;9.5;-5.0;2367.0964;67.55065;1
969700;1070.0;0.9;9.5;-5.0;2366.6143;67.55169;1
969800;1068.75;0.9;9.5;-5.0;2366.1343;67.55277;1
969900;1069.75;0.9;9.5;-5.0;2365.656;67.55389;1
970000;1070.75;0.9;9.5;-5.0;2365.1797;67.55504;1
970100;1069.25;0.9;9.5;-5.0;2364.7053;67.55624;1
970200;1069.25;0.9;9.5;-5.0;2364.2327;67.55747;1
970300;1069.0;0.9;9.5;-5.0;2363.762;67.55875;1
970400;1071.25;0.9;9.5;-5.0;2363.2932;67.56006;1
970500;1070.75;0.9;9.5;-5.0;2362.8262;67.56141;1
970600;1072.75;0.9;9.5;-5.0;2362.361;67.5628;1
970700;1071.25;0.9;9.5;-5.0;2361.898;67.564224;1
970800;1070.25;0.9;9.5;-5.0;2361.4365;67.5657;1
970900;1072.25;0.9;9.5;-5.0;2360.977;67.56721;1
971000;1070.75;0.9;9.5;-5.0;2360.5195;67.568756;1
971100;1071.0;0.9;9.5;-5.0;2360.0637;67.57034;1
971200;1068.75;0.9;9.5;-5.0;2359.6099;67.571976;1
971300;1067.75;0.9;9.5;-5.0;2359.158;67.57364;1
971400;1066.75;0.9;9.5;-5.0;2358.7078;67.57535;1
971500;1067.25;0.9;9.5;-5.0;2358.2595;67.577095;1
971600;1067.25;0.9;9.5;-5.0;2357.813;67.57889;1
971700;1066.25;0.9;9.5;-5.0;2357.3684;67.58072;1
971800;1066.75;0.9;9.5;-5.0;2356.9258;67.58259;1
971900;1067.75;0.9;9.5;-5.0;2356.4849;67.584496;1
972000;1066.75;0.9;9.5;-5.0;2356.0457;67.58644;1
972100;1066.25;0.9;9.5;-5.0;2355.6086;67.58843;1
972200;1067.75;0.9;9.5;-5.0;2355.1733;67.59046;1
972300;1067.25;0.9;9.5;-5.0;2354.7397;67.59254;1
972400;1065.75;0.9;9.5;-5.0;2354.308;67.59465;1
972500;1066.75;0.9;9.5;-5.0;2353.8784;67.5968;1
972600;1065.75;0.9;9.5;-5.0;2353.4504;67.599;1
972700;1063.75;0.9;9.5;-5.0;2353.0242;67.601234;1
972800;1061.75;0.9;9.5;-5.0;2352.5999;67.60351;1
972900;1062.0;0.9;9.5;-5.0;2352.1775;67.60583;1
973000;1060.75;0.9;9.5;-5.0;2351.7568;67.608185;1
973100;1063.25;0.9;9.5;-5.0;2351.3381;67.61058;1
973200;1065.25;0.9;9.5;-5.0;2350.9211;67.61302;1
973300;1065.25;0.9;9.5;-5.0;2350.506;67.6155;1
973400;1066.25;0.9;9.5;-5.0;2350.0928;67.61803;1
973500;1066.0;0.9;9.5;-5.0;2349.6812;67.62059;1
973600;1066.75;0.9;9.5;-5.0;2349.2717;67.6232;1
973700;1066.75;0.9;9.5;-5.0;2348.8638;67.62585;1
973800;1064.75;0.9;9.5;-5.0;2348.4578;67.62854;1
973900;1063.75;0.9;9.5;-5.0;2348.0537;67.63127;1
974000;1064.25;0.9;9.5;-5.0;2347.6514;67.63405;1
974100;1064.5;0.9;9.5;-5.0;2347.2507;67.63686;1
974200;1064.5;0.9;9.5;-5.0;2346.852;67.63972;1
974300;1066.25;0.9;9.5;-5.0;2346.455;67.642624;1
974400;1067.25;0.9;9.5;-5.0;2346.06;67.64556;1
974500;1065.25;0.9;9.5;-5.0;2345.6667;67.648544;1
974600;1063.75;0.9;9.5;-5.0;2345.2751;67.65157;1
974700;1064.75;0.9;9.5;-5.0;2344.8855;67.65465;1
974800;1062.75;0.9;9.5;-5.0;2344.4976;67.65775;1
974900;1062.25;0.9;9.5;-5.0;2344.1116;67.66091;1
975000;1059.25;0.9;9.5;-5.0;2343.7273;67.66411;1
975100;1060.25;0.9;9.5;-5.0;2343.345;67.66735;1
975200;1060.25;0.9;9.5;-5.0;2342.964;67.67063;1
975300;1060.25;0.9;9.5;-5.0;2342.5854;67.67396;1
975400;1060.25;0.9;9.5;-5.0;2342.2083;67.67733;1
975500;1058.25;0.9;9.5;-5.0;2341.833;67.68074;1
975600;1056.25;0.9;9.5;-5.0;2341.4595;67.6842;1
975700;1056.75;0.9;9.5;-5.0;2341.088;67.68769;1
975800;1053.75;0.9;9.5;-5.0;2340.718;67.69123;1
975900;1052.25;0.9;9.5;-5.0;2340.3499;67.69482;1
976000;1051.25;0.9;9.5;-5.0;2339.9834;67.69845;1
976100;1052.75;0.9;9.5;-5.0;2339.619;67.70212;1
976200;1054.75;0.9;9.5;-5.0;2339.256;67.70583;1
976300;1055.75;0.9;9.5;-5.0;2338.8953;67.709595;1
976400;1056.25;0.9;9.5;-5.0;2338.536;67.713394;1
976500;1057.25;0.9;9.5;-5.0;2338.1785;67.71724;1
976600;1055.75;0.9;9.5;-5.0;2337.823;67.72113;1
976700;1054.75;0.9;9.5;-5.0;2337.469;67.72507;1
976800;1057.75;0.9;9.5;-5.0;2337.117;67.72904;1
976900;1059.25;0.9;9.5;-5.0;2336.7666;67.73307;1
977000;1058.25;0.9;9.5;-5.0;2336.418;67.73714;1
977100;1058.75;0.9;9.5;-5.0;2336.0713;67.74125;1
977200;1056.75;0.9;9.5;-5.0;2335.7263;67.7454;1
977300;1058.25;0.9;9.5;-5.0;2335.383;67.7496;1
977400;1058.5;0.9;9.5;-5.0;2335.0415;67.753845;1
977500;1060.25;0.9;9.5;-5.0;2334.702;67.75813;1
977600;1061.75;0.9;9.5;-5.0;2334.3638;67.76247;1
977700;1060.75;0.9;9.5;-5.0;2334.0276;67.766846;1
977800;1061.75;0.9;9.5;-5.0;2333.693;67.77127;1
977900;1061.25;0.9;9.5;-5.0;2333.3606;67.775734;1
978000;1060.75;0.9;9.5;-5.0;2333.0295;67.78025;1
978100;1062.75;0.9;9.5;-5.0;2332.7004;67.784805;1
978200;1060.25;0.9;9.5;-5.0;2332.373;67.789406;1
978300;1059.25;0.9;9.5;-5.0;2332.0474;67.79405;1
978400;1058.75;0.9;9.5;-5.0;2331.7234;67.79875;1
978500;1057.25;0.9;9.5;-5.0;2331.4011;67.80349;1
978600;1058.25;0.9;9.5;-5.0;2331.0808;67.80827;1
978700;1055.75;0.9;9.5;-5.0;2330.762;67.8131;1
978800;1055.75;0.9;9.5;-5.0;2330.445;67.81797;1
978900;1056.75;0.9;9.5;-5.0;2330.13;67.82289;1
979000;1057.0;0.9;9.5;-5.0;2329.8164;67.82786;1
979100;1057.0;0.9;9.5;-5.0;2329.5046;67.83287;1
979200;1057.25;0.9;9.5;-5.0;2329.1946;67.83793;1
979300;1057.5;0.9;9.5;-5.0;2328.8862;67.84303;1
979400;1058.75;0.9;9.5;-5.0;2328.5798;67.84818;1
979500;1058.25;0.9;9.5;-5.0;2328.275;67.85338;1
979600;1056.75;0.9;9.5;-5.0;2327.972;67.85862;1
979700;1055.75;0.9;9.5;-5.0;2327.6704;67.86391;1
979800;1055.75;0.9;9.5;-5.0;2327.3708;67.86924;1
979900;1054.25;0.9;9.5;-5.0;2327.073;67.87462;1
980000;1054.5;0.9;9.5;-5.0;2326.7769;67.88005;1
980100;1055.25;0.9;9.5;-5.0;2326.4824;67.88552;1
980200;1055.5;0.9;9.5;-5.0;2326.1897;67.89104;1
980300;1054.25;0.9;9.5;-5.0;2325.8984;67.89661;1
980400;1053.75;0.9;9.5;-5.0;2325.6091;67.90222;1
980500;1052.75;0.9;9.5;-5.0;2325.3215;67.90788;1
980600;1050.75;0.9;9.5;-5.0;2325.0356;67.91359;1
980700;1050.25;0.9;9.5;-5.0;2324.7517;67.91934;1
980800;1049.75;0.9;9.5;-5.0;2324.4692;67.92514;1
980900;1049.75;0.9;9.5;-5.0;2324.1885;67.93099;1
981000;1050.75;0.9;9.5;-5.0;2323.9094;67.93688;1
981100;1052.25;0.9;9.5;-5.0;2323.632;67.942825;1
981200;1050.75;0.9;9.5;-5.0;2323.3564;67.948814;1
981300;1051.75;0.9;9.5;-5.0;2323.0825;67.95486;1
981400;1050.75;0.9;9.5;-5.0;2322.81;67.96094;1
981500;1052.25;0.9;9.5;-5.0;2322.5396;67.96707;1
981600;1051.75;0.9;9.5;-5.0;2322.2708;67.97325;1
981700;1052.25;0.9;9.5;-5.0;2322.0037;67.97948;1
981800;1052.75;0.9;9.5;-5.0;2321.7383;67.985756;1
981900;1053.25;0.9;9.5;-5.0;2321.4744;67.99207;1
982000;1052.25;0.9;9.5;-5.0;2321.2124;67.99844;1
982100;1052.0;0.9;9.5;-5.0;2320.952;68.00487;1
982200;1048.25;0.9;9.5;-5.0;2320.6934;68.01133;1
982300;1048.5;0.9;9.5;-5.0;2320.4363;68.017845;1
982400;1049.75;0.9;9.5;-5.0;2320.181;68.024414;1
982500;1049.75;0.9;9.5;-5.0;2319.9272;68.03102;1
982600;1051.25;0.9;9.5;-5.0;2319.6753;68.03768;1
982700;1048.25;0.9;9.5;-5.0;2319.425;68.044395;1
982800;1049.25;0.9;9.5;-5.0;2319.1765;68.05115;1
982900;1049.75;0.9;9.5;-5.0;2318.9294;68.05795;1
983000;1050.0;0.9;9.5;-5.0;2318.6843;68.06481;1
983100;1050.75;0.9;9.5;-5.0;2318.4407;68.07171;1
983200;1052.25;0.9;9.5;-5.0;2318.1987;68.07867;1
983300;1050.75;0.9;9.5;-5.0;2317.9585;68.08566;1
983400;1052.25;0.9;9.5;-5.0;2317.72;68.09271;1
983500;1051.75;0.9;9.5;-5.0;2317.4832;68.099815;1
983600;1051.25;0.9;9.5;-5.0;2317.2478;68.106964;1
983700;1051.75;0.9;9.5;-5.0;2317.0142;68.11416;1
983800;1053.75;0.9;9.5;-5.0;2316.7825;68.12141;1
983900;1052.25;0.9;9.5;-5.0;2316.552;68.1287;1
984000;1054.75;0.9;9.5;-5.0;2316.3235;68.13605;1
984100;1054.75;0.9;9.5;-5.0;2316.0967;68.14344;1
984200;1052.25;0.9;9.5;-5.0;2315.8713;68.15089;1
984300;1051.75;0.9;9.5;-5.0;2315.6477;68.15838;1
984400;1051.75;0.9;9.5;-5.0;2315.4258;68.165924;1
984500;1053.25;0.9;9.5;-5.0;2315.2053;68.173515;1
984600;1053.25;0.9;9.5;-5.0;2314.9866;68.18116;1
984700;1054.25;0.9;9.5;-5.0;2314.7695;68.18885;1
984800;1055.25;0.9;9.5;-5.0;2314.5542;68.196594;1
984900;1056.75;0.9;9.5;-5.0;2314.3406;68.20439;1
985000;1056.25;0.9;9.5;-5.0;2314.1284;68.212234;1
985100;1054.75;0.9;9.5;-5.0;2313.918;68.22012;1
985200;1055.75;0.9;9.5;-5.0;2313.7092;68.22807;1
985300;1054.25;0.9;9.5;-5.0;2313.502;68.23606;1
985400;1053.75;0.9;9.5;-5.0;2313.2964;68.24411;1
985500;1054.75;0.9;9.5;-5.0;2313.0925;68.252205;1
985600;1052.75;0.9;9.5;-5.0;2312.8901;68.26035;1
985700;1053.25;0.9;9.5;-5.0;2312.6897;68.26855;1
985800;1054.25;0.9;9.5;-5.0;2312.4905;68.276794;1
985900;1054.25;0.9;9.5;-5.0;2312.2932;68.285095;1
986000;1056.25;0.9;9.5;-5.0;2312.0974;68.29344;1
986100;1057.25;0.9;9.5;-5.0;2311.9033;68.30184;1
986200;1060.25;0.9;9.5;-5.0;2311.7107;68.310295;1
986300;1061.75;0.9;9.5;-5.0;2311.52;68.318794;1
986400;1062.75;0.9;9.5;-5.0;2311.3306;68.327354;1
986500;1062.75;0.9;9.5;-5.0;2311.143;68.33595;1
986600;1063.75;0.9;9.5;-5.0;2310.957;68.34461;1
986700;1062.25;0.9;9.5;-5.0;2310.7725;68.35332;1
986800;1065.25;0.9;9.5;-5.0;2310.5898;68.36208;1
986900;1066.75;0.9;9.5;-5.0;2310.4084;68.37089;1
987000;1067.25;0.9;9.5;-5.0;2310.229;68.37975;1
987100;1067.5;0.9;9.5;-5.0;2310.051;68.38867;1
987200;1069.25;0.9;9.5;-5.0;2309.8745;68.39764;1
987300;1069.25;0.9;9.5;-5.0;2309.7;68.406654;1
987400;1068.75;0.9;9.5;-5.0;2309.5266;68.415726;1
987500;1066.75;0.9;9.5;-5.0;2309.3552;68.42484;1
987600;1068.25;0.9;9.5;-5.0;2309.1853;68.43402;1
987700;1069.25;0.9;9.5;-5.0;2309.0168;68.443245;1
987800;1070.75;0.9;9.5;-5.0;2308.85;68.45252;1
987900;1069.75;0.9;9.5;-5.0;2308.685;68.46185;1
988000;1071.25;0.9;9.5;-5.0;2308.5215;68.47124;1
988100;1069.75;0.9;9.5;-5.0;2308.3594;68.480675;1
988200;1069.25;0.9;9.5;-5.0;2308.199;68.490166;1
988300;1070.25;0.9;9.5;-5.0;2308.0403;68.4997;1
988400;1068.75;0.9;9.5;-5.0;2307.883;68.5093;1
988500;1067.25;0.9;9.5;-5.0;2307.7275;68.51894;1
988600;1068.25;0.9;9.5;-5.0;2307.5735;68.52865;1
988700;1068.75;0.9;9.5;-5.0;2307.421;68.5384;1
988800;1068.5;0.9;9.5;-5.0;2307.27;68.5482;1
988900;1068.5;0.9;9.5;-5.0;2307.1208;68.55806;1
989000;1068.25;0.9;9.5;-5.0;2306.9731;68.56798;1
989100;1068.75;0.9;9.5;-5.0;2306.827;68.57794;1
989200;1068.75;0.9;9.5;-5.0;2306.6824;68.58796;1
989300;1068.25;0.9;9.5;-5.0;2306.5396;68.59803;1
989400;1068.25;0.9;9.5;-5.0;2306.3982;68.608154;1
989500;1068.25;0.9;9.5;-5.0;2306.2583;68.61833;1
989600;1066.25;0.9;9.5;-5.0;2306.1199;68.62857;1
989700;1065.75;0.9;9.5;-5.0;2305.9834;68.638855;1
989800;1066.25;0.9;9.5;-5.0;2305.8481;68.64919;1
989900;1066.75;0.9;9.5;-5.0;2305.7146;68.65959;1
990000;1063.75;0.9;9.5;-5.0;2305.5825;68.67004;1
990100;1064.75;0.9;9.5;-5.0;2305.4521;68.680534;1
990200;1063.25;0.9;9.5;-5.0;2305.3232;68.69109;1
990300;1063.25;0.9;9.5;-5.0;2305.196;68.701706;1
990400;1061.75;0.9;9.5;-5.0;2305.07;68.712364;1
990500;1060.75;0.9;9.5;-5.0;2304.946;68.72308;1
990600;1059.75;0.9;9.5;-5.0;2304.8232;68.73386;1
990700;1057.25;0.9;9.5;-5.0;2304.7021;68.74469;1
990800;1058.25;0.9;9.5;-5.0;2304.5825;68.75557;1
990900;1059.75;0.9;9.5;-5.0;2304.4646;68.7665;1
991000;1057.75;0.9;9.5;-5.0;2304.3481;68.7775;1
991100;1056.75;0.9;9.5;-5.0;2304.2332;68.78854;1
991200;1056.75;0.9;9.5;-5.0;2304.1199;68.799644;1
991300;1055.25;0.9;9.5;-5.0;2304.008;68.8108;1
991400;1055.5;0.9;9.5;-5.0;2303.8977;68.82201;1
991500;1056.25;0.9;9.5;-5.0;2303.789;68.833275;1
991600;1055.25;0.9;9.5;-5.0;2303.682;68.8446;1
991700;1053.25;0.9;9.5;-5.0;2303.5762;68.85597;1
991800;1053.5;0.9;9.5;-5.0;2303.472;68.8674;1
991900;1054.25;0.9;9.5;-5.0;2303.3694;68.87889;1
992000;1054.0;0.9;9.5;-5.0;2303.2683;68.890434;1
992100;1055.25;0.9;9.5;-5.0;2303.1687;68.90203;1
992200;1055.25;0.9;9.5;-5.0;2303.0706;68.91368;1
992300;1054.75;0.9;9.5;-5.0;2302.974;68.92539;1
992400;1056.25;0.9;9.5;-5.0;2302.8792;68.93716;1
992500;1056.25;0.9;9.5;-5.0;2302.7856;68.948975;1
992600;1056.75;0.9;9.5;-5.0;2302.6938;68.96085;1
992700;1056.25;0.9;9.5;-5.0;2302.6035;68.972786;1
992800;1057.75;0.9;9.5;-5.0;2302.5144;68.98477;1
992900;1057.25;0.9;9.5;-5.0;2302.4272;68.99682;1
993000;1055.75;0.9;9.5;-5.0;2302.3413;69.00892;1
993100;1056.75;0.9;9.5;-5.0;2302.2568;69.02107;1
993200;1055.75;0.9;9.5;-5.0;2302.174;69.03329;1
993300;1053.75;0.9;9.5;-5.0;2302.0928;69.045555;1
993400;1052.25;0.9;9.5;-5.0;2302.013;69.05788;1
993500;1052.25;0.9;9.5;-5.0;2301.9346;69.07026;1
993600;1052.75;0.9;9.5;-5.0;2301.858;69.0827;1
993700;1051.25;0.9;9.5;-5.0;2301.7825;69.0952;1
993800;1052.25;0.9;9.5;-5.0;2301.7087;69.10775;1
993900;1051.25;0.9;9.5;-5.0;2301.6365;69.12036;1
994000;1050.75;0.9;9.5;-5.0;2301.5657;69.133026;1
994100;1049.25;0.9;9.5;-5.0;2301.4963;69.145744;1
994200;1049.25;0.9;9.5;-5.0;2301.4285;69.15853;1
994300;1052.25;0.9;9.5;-5.0;2301.362;69.171364;1
994400;1050.75;0.9;9.5;-5.0;2301.2974;69.18426;1
994500;1051.75;0.9;9.5;-5.0;2301.2341;69.19721;1
994600;1051.75;0.9;9.5;-5.0;2301.172;69.21022;1
994700;1052.0;0.9;9.5;-5.0;2301.1118;69.22329;1
994800;1049.75;0.9;9.5;-5.0;2301.053;69.23641;1
994900;1049.75;0.9;9.5;-5.0;2300.9956;69.249596;1
995000;1048.25;0.9;9.5;-5.0;2300.9397;69.26283;1
995100;1050.75;0.9;9.5;-5.0;2300.8853;69.27613;1
995200;1050.5;0.9;9.5;-5.0;2300.8323;69.28949;1
995300;1049.75;0.9;9.5;-5.0;2300.7808;69.3029;1
995400;1050.75;0.9;9.5;-5.0;2300.731;69.31637;1
995500;1051.25;0.9;9.5;-5.0;2300.6824;69.3299;1
995600;1051.0;0.9;9.5;-5.0;2300.6353;69.34348;1
995700;1052.75;0.9;9.5;-5.0;2300.5898;69.35713;1
995800;1052.5;0.9;9.5;-5.0;2300.5457;69.370834;1
995900;1051.75;0.9;9.5;-5.0;2300.503;69.3846;1
996000;1052.75;0.9;9.5;-5.0;2300.462;69.398415;1
996100;1050.25;0.9;9.5;-5.0;2300.422;69.4123;1
996200;1048.75;0.9;9.5;-5.0;2300.384;69.42623;1
996300;1049.75;0.9;9.5;-5.0;2300.3472;69.44023;1
996400;1048.25;0.9;9.5;-5.0;2300.3118;69.454285;1
996500;1049.75;0.9;9.5;-5.0;2300.278;69.4684;1
996600;1051.75;0.9;9.5;-5.0;2300.2456;69.482574;1
996700;1053.75;0.9;9.5;-5.0;2300.2146;69.4968;1
996800;1054.75;0.9;9.5;-5.0;2300.1853;69.51109;1
996900;1055.0;0.9;9.5;-5.0;2300.1572;69.525444;1
997000;1054.25;0.9;9.5;-5.0;2300.1306;69.53985;1
997100;1055.25;0.9;9.5;-5.0;2300.1055;69.55432;1
997200;1052.75;0.9;9.5;-5.0;2300.0818;69.56885;1
997300;1051.75;0.9;9.5;-5.0;2300.0596;69.583435;1
997400;1049.75;0.9;9.5;-5.0;2300.0388;69.598076;1
997500;1049.75;0.9;9.5;-5.0;2300.0193;69.612785;1
997600;1049.25;0.9;9.5;-5.0;2300.0015;69.62755;1
997700;1049.75;0.9;9.5;-5.0;2299.985;69.64237;1
997800;1050.75;0.9;9.5;-5.0;2299.97;69.65726;1
997900;1052.75;0.9;9.5;-5.0;2299.9563;69.6722;1
998000;1050.25;0.9;9.5;-5.0;2299.9443;69.68721;1
998100;1048.75;0.9;9.5;-5.0;2299.9336;69.70227;1
998200;1050.25;0.9;9.5;-5.0;2299.9243;69.7174;1
998300;1050.25;0.9;9.5;-5.0;2299.9163;69.73258;1
998400;1049.75;0.9;9.5;-5.0;2299.91;69.747826;1
998500;1050.0;0.9;9.5;-5.0;2299.905;69.76313;1
998600;1048.75;0.9;9.5;-5.0;2299.9014;69.778496;1
998700;1048.25;0.9;9.5;-5.0;2299.8992;69.79392;1
998800;1045.25;0.9;9.5;-5.0;2299.8984;69.80941;1
998900;1044.25;0.9;9.5;-5.0;2299.8992;69.82496;1
999000;1044.25;0.9;9.5;-5.0;2299.9014;69.84057;1
999100;1042.75;0.9;9.5;-5.0;2299.9048;69.85624;1
999200;1040.75;0.9;9.5;-5.0;2299.9097;69.87196;1
999300;1041.75;0.9;9.5;-5.0;2299.916;69.88776;1
999400;1039.75;0.9;9.5;-5.0;2299.9238;69.90361;1
999500;1038.75;0.9;9.5;-5.0;2299.933;69.91952;1
999600;1036.25;0.9;9.5;-5.0;2299.9436;69.93549;1
999700;1036.25;0.9;9.5;-5.0;2299.9556;69.95153;1
999800;1036.25;0.9;9.5;-5.0;2299.969;69.96762;1
999900;1034.25;0.9;9.5;-5.0;2299.984;69.98378;1
1000000;1035.25;0.9;9.5;-5.0;2300.0;70.0;1"""