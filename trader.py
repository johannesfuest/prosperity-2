import json
import math
import string
from typing import List

import numpy as np

from datamodel import Order, OrderDepth, TradingState, UserId


class Trader:

    def __init__(self):
        self.limits = {
            "AMETHYSTS": 20,
            "STARFRUIT": 20,
        } 
        
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
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        self.print_input_state(state)
        print("Start Evaluation:")
        print("=================\n")
        result = {}
        traderData = {}
        buy_thresholds = {}
        sell_thresholds = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = get_acceptable_price_for_product(state, product)
            buy_spread = get_product_edge(state, product, "buy")
            sell_spread = get_product_edge(state, product, "sell")
            print("## Acceptable price : " + str(acceptable_price))
            print("## We Buy below : " + str(acceptable_price * (1 - buy_spread)) + " and sell above : " + str(acceptable_price * (1 + sell_spread)))
            print("## Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            if product == 'AMETHYSTS':
                buy_thresholds[product] = int(np.floor(acceptable_price * (1 - buy_spread)))
                sell_thresholds[product] = int(np.ceil(acceptable_price * (1 + sell_spread)))
            else:
                buy_thresholds[product] = int(np.floor(acceptable_price * (1 - 0.5*buy_spread)))
                sell_thresholds[product] = int(np.ceil(acceptable_price * (1 + 1.5*sell_spread)))
            if len(order_depth.sell_orders) != 0:
                for ask, ask_amount in order_depth.sell_orders.items():
                    if int(ask) < acceptable_price * (1 - (buy_spread)):
                        print("## BUY", str(-ask_amount) + "x", ask)
                        orders.append(Order(product, math.floor(ask), -ask_amount))
                
    
            if len(order_depth.buy_orders) != 0:
                for bid, bid_amount in order_depth.buy_orders.items():
                    if int(bid) > acceptable_price * (1 + (sell_spread)):
                        print("## SELL", str(bid_amount) + "x", bid)
                        orders.append(Order(product, math.ceil(bid), -bid_amount))
            
            result[product] = orders
            traderData[product] = generate_trader_data(state, product)
            
        result = self.adjust_for_position_breaches(result, state, True)
        result = self.adjust_to_exploit_limits(
            result, state, buy_thresholds, sell_thresholds)
        traderData = json.dumps(traderData)
        conversions = 1
        return result, conversions, traderData
    
    def adjust_for_position_breaches(self, results, state, fill_until_position_breach=False):
        valid_orders = {}
        for product, orders in results.items():
            valid_orders[product] = []
            buy_orders = [order for order in orders if order.quantity > 0]
            buy_orders = sorted(buy_orders, key=lambda x: x.price)
            sell_orders = [order for order in orders if order.quantity < 0]
            sell_orders = sorted(sell_orders, key=lambda x: x.price, reverse=True)
            cur_position = state.position.get(product, 0)
            tmp_cur_position = cur_position
            for buy_order in buy_orders:
                if tmp_cur_position + buy_order.quantity > 20:
                    if fill_until_position_breach:
                        buy_order = Order(product, buy_order.price, 20 - tmp_cur_position)
                        valid_orders[product].append(buy_order)
                        break
                else:
                    valid_orders[product].append(buy_order)
                    tmp_cur_position += buy_order.quantity
            tmp_cur_position = cur_position
            for sell_order in sell_orders:
                if tmp_cur_position + sell_order.quantity < -20:
                    if fill_until_position_breach:
                        sell_order = Order(product, sell_order.price, -20 - tmp_cur_position)
                        valid_orders[product].append(sell_order)
                        tmp_cur_position += sell_order.quantity
                        break
                else:
                    valid_orders[product].append(sell_order)
                    tmp_cur_position += sell_order.quantity
        return valid_orders
    
    def adjust_to_exploit_limits(self, results, state, buy_threshold, sell_threshold):
        for product, orders, in results.items():
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
                            results[product].append(Order(product, sell_threshold[product] + 1 + i, -n))
                    sell_order = Order(product, sell_threshold[product] + 1, -numbers[0])
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
                            results[product].append(Order(product, buy_threshold[product] - 1 - i, n))
                    buy_order = Order(product, buy_threshold[product] - 1, numbers[0])
                results[product].append(buy_order)
        return results
            
    
def get_product_edge(state, product, buy_sell):
    price_history = get_price_history_from_state(state, product)
    if product == "AMETHYSTS":
        return 0.00015/2
    elif product == "STARFRUIT":
        if buy_sell == "buy":
            return (np.std(price_history) / np.mean(price_history)) / 2
        else:
            return (np.std(price_history) / np.mean(price_history)) / 2
    

def generate_trader_data(state, product):
    price_history_list = get_price_history_from_state(state, product)
    price_history_list.pop(0)
    price_history_list.append(get_mid_price_from_order_book(state.order_depths, product))
    return price_history_list
        

def initialize_trader_data(product, length=15):

    match product:
        case "AMETHYSTS":
            inital_data =  [
                9999.0,
                10000.0,
                10000.0,
                10003.5,
                9999.0,
                10003.5,
                10001.0,
                10000.0,
                10000.0,
                10000.0,
                9998.5,
                9999.0,
                10000.0,
                10000.0,
                10000.0,
            ]
        
        case "STARFRUIT":
            inital_data = [
                5053.0,
                5053.5,
                5052.5,
                5053.5,
                5053.0,
                5053.0,
                5052.0,
                5051.5,
                5052.0,
                5051.5,
                5052.5,
                5051.0,
                5053.5,
                5049.5,
                5051.0,
            ]
    return inital_data[-length:]
    
    

def get_acceptable_price_for_product(state, product):

    coefs = np.array([
        1.7044926379649041,
        0.2920955,
        0.20671938,
        0.14077617,
        0.10025522,
        0.08580541 ,
        0.06038695,
        0.03888277,
        0.00594952,
        0.02262225,
        0.01394354,
        0.0164973,
        0.00535559,
        0.00513494,
        0.00572899,
        -0.00049075
        ])
    
    price_history = get_price_history_from_state(state, product)

    if product == "AMETHYSTS":
        return sum(price_history) / len(price_history)
    else:
        price_history = np.array([1.0] + list(reversed(price_history)))
        predicted_price = np.dot(coefs, price_history)
    return predicted_price
    

    
def get_mid_price_from_order_book(order_depth, product):    
    best_bid = list(order_depth[product].buy_orders.keys())[0]
    best_ask = list(order_depth[product].sell_orders.keys())[0]
    return (best_bid + best_ask) / 2

def get_price_history_from_state(state, product):
    if not state.traderData:
        return initialize_trader_data(product)
    else:
        return json.loads(state.traderData)[product]

def split_number(n):
    # Calculate the first part
    first = n // 6
    
    # Calculate the second part, approximately twice the first
    second = 2 * first
    
    # Calculate the third part, approximately three times the first
    third = 3 * first
    
    # Correct for any rounding errors by adjusting the third part
    third += n - (first + second + third)
    
    return first, second, third