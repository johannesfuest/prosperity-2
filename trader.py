import string
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
        for product in state.order_depths:
            print(f"# Product: {product}")
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = get_acceptable_price_for_product(product, state)
            print("## Acceptable price : " + str(acceptable_price))
            print("## Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            spread = 0.001 if product == "STARFRUIT" else 0.01
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price * (1 - spread):
                    print("## BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price *(1 + spread):
                    print("## SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
        print(f"Positions{state.position}")
            
        orders = self.adjust_for_position_breaches(result, state, True)
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
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
        return orders

<<<<<<< HEAD
def get_acceptable_price_for_product(product, state, strategy=None):
=======
def get_acceptable_price_for_product(product, state):
>>>>>>> 103819b (added stuff and rebased)
    product_trade_history = state.market_trades.get(product, [])
    product_order_depth = state.order_depths[product]
    if len(product_trade_history) == 0:
        return get_mid_price_from_order_book(product_order_depth)
    else:
        return product_trade_history[0].price
    
def get_mid_price_from_order_book(order_depth):
    best_bid = list(order_depth.buy_orders.keys())[0]
    best_ask = list(order_depth.sell_orders.keys())[0]
    return (best_bid + best_ask) / 2
        