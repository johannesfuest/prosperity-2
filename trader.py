import json
import math
import string
from typing import List

import numpy as np

from datamodel import Order, OrderDepth, TradingState, UserId


class Trader:

    def __init__(self):
        # 1 gift basket = 6 strawberries, four chocolates and one rose
        self.limits = {
            "AMETHYSTS": 20,
            "STARFRUIT": 20,
            "ORCHIDS": 100,
            "CHOCOLATE": 250,
            "STRAWBERRIES": 350,
            "ROSES": 60,
            "GIFT_BASKET": 60,
        } 
    
    
    def run(self, state: TradingState):
        result = {}
        traderData = {}
        buy_thresholds = {}
        sell_thresholds = {}
        conversions = None
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = self.get_acceptable_price_for_product(state, product)
            product_vol = get_product_vol(state, product)
            vol_factor_buy = {
                "AMETHYSTS": 1,
                "STARFRUIT": 0.25,
                "ORCHIDS": 1000,
                "CHOCOLATE": 1,
                "STRAWBERRIES": 1,
                "ROSES": 1,
                "GIFT_BASKET": 1,
            }
            vol_factor_sell = {
                "AMETHYSTS": 1,
                "STARFRUIT": 0.25,
                "ORCHIDS": 1000,
                "CHOCOLATE": 1,
                "STRAWBERRIES": 1,
                "ROSES": 1,
                "GIFT_BASKET": 1,
            }
            buy_thres = int(np.floor(acceptable_price * (1 - vol_factor_buy[product] * product_vol)))
            sell_thresh = int(np.ceil(acceptable_price * (1 + vol_factor_sell[product] * product_vol)))
            buy_thresholds[product] = buy_thres
            sell_thresholds[product] = sell_thresh
            if product == "ORCHIDS":
                orders, conversions = self.get_orchid_trades(state)
                result[product] = orders
            else:
                if len(order_depth.sell_orders) != 0:
                    for ask, ask_amount in order_depth.sell_orders.items():
                        if int(ask) <= buy_thresholds[product]:
                            print("## BUY", str(-ask_amount) + "x", ask)
                            orders.append(Order(product, math.floor(ask), -ask_amount))
        
                if len(order_depth.buy_orders) != 0:
                    for bid, bid_amount in order_depth.buy_orders.items():
                        if int(bid) >= sell_thresholds[product]:
                            print("## SELL", str(bid_amount) + "x", bid)
                            orders.append(Order(product, math.ceil(bid), -bid_amount))
            result[product] = orders
            traderData[product] = update_price_history(state, product)
        
        result = self.adjust_for_position_breaches(result, state, True)
        result = self.adjust_to_exploit_limits(
            result, state, buy_thresholds, sell_thresholds)
        traderData = json.dumps(traderData)
        
        #TODO: add something to the trader data that allows us to check profit from conversions of orchids
        return result, conversions, traderData

    def get_orchid_trades(self, state):
        orchid_data = state.observations.conversionObservations['ORCHIDS']

        bid_price_south = orchid_data.bidPrice
        ask_price_south = orchid_data.askPrice

        # adjust south bid and ask for transport fees
        bid_price_south = bid_price_south - orchid_data.exportTariff - orchid_data.transportFees
        ask_price_south = ask_price_south + orchid_data.importTariff + orchid_data.transportFees

        # Ship off all previous round orchids
        conversion = abs(self.get_position("ORCHIDS", state))
        orders = []
        # positive numbers of how many we can buy and sell without hitting the limit
        max_buy_capacity = self.limits['ORCHIDS'] - self.get_position("ORCHIDS", state)
        max_sell_capacity = self.limits['ORCHIDS'] + self.get_position("ORCHIDS", state)

        # get north 
        north_best_bid, north_best_bid_amount = self.get_best_bid("ORCHIDS", state)
        north_best_ask, north_best_ask_amount = self.get_best_ask("ORCHIDS", state)
        # adjust for holding costs
        #Johannes: this should be a plus, since we hold stuff we buy for a round
        north_best_ask = north_best_ask + 0.1 * north_best_ask_amount
        
        expected_profit_dict = {}
        if (north_best_bid > ask_price_south) & (max_sell_capacity > 0):
            expected_profit_dict[(north_best_bid, -north_best_bid_amount)] = (north_best_bid - ask_price_south)

        if (north_best_ask < bid_price_south) & (max_buy_capacity > 0):
            orders.append(Order("ORCHIDS", north_best_ask, north_best_ask_amount))
            expected_profit_dict[(north_best_ask, north_best_ask_amount)] = (bid_price_south - north_best_ask)

        for i in range(1, 10):
            bid, bid_amount = self.get_best_bid("ORCHIDS", state, i)
            ask, ask_amount = self.get_best_ask("ORCHIDS", state, i)
            if bid is not None:
                if (bid > ask_price_south) & (max_sell_capacity > 0):
                    expected_profit_dict[(bid, -bid_amount)] = (bid - ask_price_south)
            if ask is not None:
                ask = ask - 0.1 * ask_amount
                if (ask < bid_price_south) & (max_buy_capacity > 0):
                    expected_profit_dict[(ask, ask_amount)] = (bid_price_south - ask)
        
        # calculate the price for orders based on the difference between the best bid/ask and the south bid/ask
        # when the difference is 1, the price is the best bid/ask
        # when the difference is 2, the price is the best bid/ask + 1
        # when the difference is 3, the price is the best bid/ask + 2.5
        def calculate_price(difference, best_price, price_south, adjustment):
            if difference <=1:
                return best_price
            elif difference <= 2:
                return price_south + adjustment * 1
            else:
                return price_south + adjustment * 2.5

        if max_sell_capacity > 0 and north_best_ask >= ask_price_south:
            difference = north_best_ask - ask_price_south
            price = calculate_price(difference, north_best_ask, ask_price_south, 1)
            expected_profit_dict[(int(math.floor(price)), -max_sell_capacity)] = price - ask_price_south

        if max_buy_capacity > 0 and north_best_bid <= bid_price_south:
            difference = bid_price_south - north_best_bid
            price = calculate_price(difference, north_best_bid, bid_price_south, -1)
            expected_profit_dict[(int(math.ceil(price)), max_buy_capacity)] = bid_price_south - price

        # sort by expected profit and submit orders
        for (price, amount), exp_profit in sorted(expected_profit_dict.items(), key=lambda x: x[1], reverse=True):
            if (exp_profit < 0) or ((max_buy_capacity <=0) and (max_sell_capacity <= 0)):
                break
            if (amount > 0) and amount>max_buy_capacity:
                amount = max_buy_capacity
            if (amount < 0) and abs(amount)>max_sell_capacity:
                amount = -max_sell_capacity
            orders.append(Order("ORCHIDS", price, amount))
            if amount > 0:
                max_buy_capacity = max_buy_capacity - amount
            if amount < 0:
                max_sell_capacity = max_sell_capacity - abs(amount)
        print(f'sum of order quantities for orchids is {sum([order.quantity for order in orders])}')
        return orders, conversion  
    
    def get_acceptable_price_for_product(self, state, product):

        star_fruit_coefs = np.array([
            #coeffs from round 1 submission
            1.7044926379649041, 0.2920955, 0.20671938, 0.14077617, 0.10025522, 0.08580541 ,0.06038695, 0.03888277, 0.00594952, 0.02262225, 0.01394354, 0.0164973, 0.00535559, 0.00513494, 0.00572899, -0.00049075
            #coeffs using only day 0 to train
            # 13.156199936551275,0.30189398,0.21454386,0.13574109,0.11238089,0.06955258,0.06800676,0.05140635,0.0071232,0.03675125
            ])
        price_history = get_price_history_from_state(state, product)
        hist_straw = get_price_history_from_state(state, "STRAWBERRIES")
        hist_choc = get_price_history_from_state(state, "CHOCOLATE")
        hist_rose = get_price_history_from_state(state, "ROSES")
        hist_basket = get_price_history_from_state(state, "GIFT_BASKET")
        synt_hist = [4*choc + 6*straw + rose for choc, straw, rose in zip(hist_choc, hist_straw, hist_rose)]
        
        avg_diff = 379.4904833333333
        running_choc_share = np.sum(hist_choc) / (4 * np.sum(synt_hist))
        running_straw_share = np.sum(hist_straw) / (6 * np.sum(synt_hist))
        running_rose_share = np.sum(hist_rose) / (np.sum(synt_hist))
        
        choc_mid = get_mid_price_from_order_book(state.order_depths, "CHOCOLATE")
        straw_mid = get_mid_price_from_order_book(state.order_depths, "STRAWBERRIES")
        rose_mid = get_mid_price_from_order_book(state.order_depths, "ROSES")
        basket_mid = get_mid_price_from_order_book(state.order_depths, "GIFT_BASKET")
        expected_rose_mid = (basket_mid - avg_diff) * running_rose_share
        expected_choc_mid = (basket_mid - avg_diff) * running_choc_share
        expected_straw_mid = (basket_mid - avg_diff) * running_straw_share
        expected_basket_mid = 4 * choc_mid + 6 * straw_mid + rose_mid + avg_diff
        match product:
            case "STARFRUIT":
                price_history = np.array([1.0] + list(reversed(price_history)))
                predicted_price = np.dot(star_fruit_coefs, price_history)
                return predicted_price
            case "CHOCOLATE":
                return expected_choc_mid
            case "STRAWBERRIES":
                return expected_straw_mid
            case "ROSES":
                return expected_rose_mid
            case "GIFT_BASKET":
                return expected_basket_mid
            case _:
                return sum(price_history) / len(price_history)
        
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    
    
    def get_best_bid(self, product, state: TradingState, index=0):
        market_bids = state.order_depths[product].buy_orders
        #best_bid = max(market_bids)
        if len(market_bids) < (index + 1):
            return None, None
        best_bid, best_bid_amount = sorted(list(market_bids.items()))[index]
        return best_bid, best_bid_amount

    def get_best_ask(self, product, state: TradingState, index=0):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        if len(market_asks) < (index + 1):
            return None, None
        best_ask, best_ask_amount = sorted(list(market_asks.items()), reverse=True)[index]

        return best_ask, best_ask_amount 
    
    def adjust_for_position_breaches(self, results, state, fill_until_position_breach=False):
        valid_orders = {}
        for product, orders in results.items():
            valid_orders[product] = []
            buy_orders = [order for order in orders if order.quantity > 0]
            buy_orders = sorted(buy_orders, key=lambda x: x.price)
            sell_orders = [order for order in orders if order.quantity < 0]
            sell_orders = sorted(sell_orders, key=lambda x: x.price, reverse=True)
            #TODO: add proper orchid handling
            if product == 'ORCHIDS':
                cur_position = 0
            else:
                cur_position = state.position.get(product, 0)
            tmp_cur_position = cur_position
            for buy_order in buy_orders:
                if tmp_cur_position + buy_order.quantity > self.limits[product]:
                    if fill_until_position_breach:
                        buy_order = Order(product, buy_order.price, self.limits[product] - tmp_cur_position)
                        valid_orders[product].append(buy_order)
                        break
                else:
                    valid_orders[product].append(buy_order)
                    tmp_cur_position += buy_order.quantity
            tmp_cur_position = cur_position
            for sell_order in sell_orders:
                if tmp_cur_position + sell_order.quantity < (-self.limits[product]):
                    if fill_until_position_breach:
                        sell_order = Order(product, sell_order.price, (-self.limits[product]) - tmp_cur_position)
                        valid_orders[product].append(sell_order)
                        tmp_cur_position += sell_order.quantity
                        break
                else:
                    valid_orders[product].append(sell_order)
                    tmp_cur_position += sell_order.quantity
        return valid_orders
    
    def adjust_to_exploit_limits(self, results, state, buy_threshold, sell_threshold):
        for product, orders, in results.items():
            if product not in ["AMETHYSTS", "STARFRUIT"]:
                continue
            cur_position = state.position.get(product, 0)
            #initally, all orders we have sent are based on the existing book and are thus guaranteed to execute
            max_pos = cur_position
            min_pos = cur_position
            for order_index, order in enumerate(orders):
                if order.quantity > 0:
                    max_pos += order.quantity
                else:
                    min_pos += order.quantity
            # If the max/min position after all orders execute is not at the limit on yet, we need to add more orders that
            # we fill the remaining quantity with optimistic trades
            n_sell_order = -min_pos - self.limits[product]
            n_buy_order = self.limits[product] - max_pos
            
            # Next we find the best remaining bid/ask after our initial orders go through and undercut if possible or

            if n_sell_order < 0:
                # our current minus all sales from this iteration is still above the limit (-20)
                found_best = False
                # we go through all the sell orders in this itertation
                for ask, ask_amount in state.order_depths[product].sell_orders.items():
                    if ask <= buy_threshold[product]:
                        # if the ask is below the buy threshold, we skip: We would have bought it already
                        continue
                    # if we land here, that means there is a sell_oder which we did not match with a buy order
                    elif ask <= sell_threshold[product]:
                        # if the ask is below our sell threshold, we can just sell at the sell threshold
                        found_best = True
                        sell_order = Order(product, sell_threshold[product], n_sell_order)
                        break
                    else:
                        # if the ask is above our sell threshold, we can just sell at the ask - 1
                        found_best = True
                        sell_order = Order(product, ask - 1, n_sell_order)
                        break
                if not found_best:
                    numbers = split_number(-n_sell_order)
                    for i, n in enumerate(numbers[1:]):
                        if n != 0:
                            results[product].append(Order(product, sell_threshold[product] + 2 + i, -n))
                    sell_order = Order(product, sell_threshold[product] + 2, -numbers[0])
                results[product].append(sell_order)
            if n_buy_order > 0:
                found_best = False
                for bid, bid_amount in state.order_depths[product].buy_orders.items():
                    if bid >= sell_threshold[product]:
                        continue
                    elif bid >= buy_threshold[product]:
                        found_best = True
                        buy_order = Order(product, buy_threshold[product], n_buy_order)
                        break
                    else:
                        found_best = True
                        buy_order = Order(product, bid + 1, n_buy_order)
                        break
                if not found_best:
                    numbers = split_number(n_buy_order)
                    for i, n in enumerate(numbers[1:]):
                        if n != 0:
                            results[product].append(Order(product, buy_threshold[product] - 2 - i, n))
                    buy_order = Order(product, buy_threshold[product] - 2, numbers[0])
                results[product].append(buy_order)
        return results
            
    
def get_product_vol(state, product):
    price_history = get_price_history_from_state(state, product)
    return (np.std(price_history) / np.mean(price_history))
    

def update_price_history(state, product):
    price_history_list = get_price_history_from_state(state, product)
    price_history_list.pop(0)
    price_history_list.append(get_mid_price_from_order_book(state.order_depths, product))
    return price_history_list
        

def initialize_trader_data(product, length=15):

    match product:
        case "AMETHYSTS":
            inital_data =  [9999.0, 10000.0, 10000.0, 10003.5, 9999.0, 10003.5, 10001.0, 10000.0, 10000.0, 10000.0, 9998.5, 9999.0, 10000.0, 10000.0, 10000.0]
        case "STARFRUIT":
            inital_data = [5053.0, 5053.5, 5052.5, 5053.5, 5053.0, 5053.0, 5052.0, 5051.5, 5052.0, 5051.5, 5052.5, 5051.0, 5053.5, 5049.5, 5051.0]
        case "ORCHIDS":
            inital_data = [1048.75,1048.25,1045.25,1044.25,1044.25,1042.75,1040.75,1041.75,1039.75,1038.75,1036.25,1036.25,1036.25,1034.25,1035.25,]
        case "CHOCOLATE":
            inital_data = [7748.5, 7749.5, 7751.5, 7752.5, 7753.0, 7754.0, 7754.5, 7752.0, 7752.0, 7752.5, 7750.5, 7750.5, 7750.5, 7750.0, 7750.0]
        case "STRAWBERRIES":
            inital_data = [3985.5, 3985.5, 3984.5, 3985.5, 3985.5, 3985.5, 3985.5, 3984.5, 3984.5, 3984.0, 3983.5, 3983.5, 3984.5, 3984.5, 3984.5]
        case "ROSES":
            inital_data = [14402.0, 14406.5, 14407.5, 14406.5, 14404.5, 14404.5, 14400.0, 14397.5, 14399.5, 14405.5, 14409.5, 14408.0, 14411.5, 14412.5, 14411.5]
        case "GIFT_BASKET":
            inital_data = [69525.5, 69537.5, 69541.5, 69558.5, 69553.0, 69564.0, 69567.5, 69545.5, 69552.0, 69548.5, 69534.5, 69529.5, 69543.0, 69542.0, 69556.0]
    return inital_data[-length:]

    
def get_mid_price_from_order_book(order_depth, product):  
    #TODO: make this failsafe   
    best_bid = list(order_depth[product].buy_orders.keys())[0]
    best_ask = list(order_depth[product].sell_orders.keys())[0]
    return (best_bid + best_ask) / 2

def get_price_history_from_state(state, product):
    if not state.traderData:
        return initialize_trader_data(product)
    else:
        return json.loads(state.traderData)[product]

def split_number(n):
    first = n // 6
    second = 2 * first
    third = 3 * first
    third += n - (first + second + third)
    return first, second, third

def print_input_state(self, state: TradingState):
        print("\nINPUT TRADING STATE")
        print("===================\n")
        print(f"# TraderData: {[str(state.traderData)]}")
        print(f"# Products: {[k for k,v in state.order_depths.items()]}")
        for product, order_depth in state.order_depths.items():
            print(f"## Orders for {product}")
            for price, volume in order_depth.buy_orders.items():
                print(f"### Buy Orders {volume}x {price}")
            for price, volume in order_depth.sell_orders.items():
                print(f"### Sell Orders {volume}x {price}")
        for product, listing in state.listings.items():
            print(f"#Listing: {listing}")
            print(f"# Listing for {product}:")
            #print(f"## Symbol: {listing.symbol}")
            #print(f"## Denomination: {listing.denomination}")
        print(f"# Own Trades: {state.own_trades}")
        print(f"# Market Trades: {state.market_trades}")
        print(f"# Position: {state.position}")
        print(f"# Observations: {str(state.observations)}")
        print("\n")