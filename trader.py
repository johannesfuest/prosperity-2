import string
import json
from typing import List

import numpy as np
from typing import List
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
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = get_acceptable_price_for_product(state, product)
            spread = get_product_spread(state, product)
            print("## Acceptable price : " + str(acceptable_price))
            print("## We Buy below : " + str(acceptable_price * (1 - spread)) + " and sell above : " + str(acceptable_price * (1 + spread)))
            print("## Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            if len(order_depth.sell_orders) != 0:
                for ask, ask_amount in order_depth.sell_orders.items():
                    if int(ask) < acceptable_price * (1 - (spread/2)):
                        print("## BUY", str(-ask_amount) + "x", ask)
                        orders.append(Order(product, ask, -ask_amount))
                
    
            if len(order_depth.buy_orders) != 0:
                for bid, bid_amount in order_depth.buy_orders.items():
                    if int(bid) > acceptable_price * (1 + (spread/2)):
                        print("## SELL", str(bid_amount) + "x", bid)
                        orders.append(Order(product, bid, -bid_amount))
            # add lowballing buy oders
            # add highballing sell orders
            # orders.append(Order(product, 500, 1))
            # orders.append(Order(product, 20000, -1))
            
            result[product] = orders
            traderData[product] = generate_trader_data(state, product)
            
        #result = self.adjust_for_position_breaches(result, state, True)
        traderData = json.dumps(traderData)
        conversions = 1
        return result, conversions, traderData
    
    def adjust_for_position_breaches(self, results, state, fill_until_position_breach=False):
        for product, orders in results.items():
            cur_position = state.position.get(product, 0)
            valid_orders = []
            for order_index, order in enumerate(orders):
                if cur_position + order.quantity <  (-self.limits[product]):
                    print(f"Short Position breach for product {product} with position {cur_position} and order {order}")
                    if fill_until_position_breach:
                        order = Order(product, order.price, (-self.limits[product]) - cur_position)
                        orders[order_index] = order
                        valid_orders.append(order)
                elif cur_position + order.quantity > self.limits[product]:
                    print(f"Long Position breach for product {product} with position {cur_position} and order {order}")
                    if fill_until_position_breach:
                        order = Order(product, order.price, self.limits[product] - cur_position)
                        orders[order_index] = order
                        valid_orders.append(order)
                else:
                    valid_orders.append(order)
                cur_position += order.quantity
            results[product] = valid_orders
        return results
    
def get_product_spread(state, product):
    price_history = get_price_history_from_state(state, product)
    if product == "AMETHYSTS":
        return 0.00015
    elif product == "STARFRUIT":
        return np.std(price_history) / np.mean(price_history)        
    

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
        price_history = np.array([1.0] + price_history)
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
        