from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:

    def print_input_state(self, state: TradingState):
        print("\nINPUT TRADING STATE")
        print("===================\n")
        print(f"# TraderData: {[str(state.traderData)]}")
        print(f"# Products: {[k for k,v in state.order_depths.items()]}")
        for product, order_depth in state.order_depths.items():
            print(f"## Oders for {product}")
            for price, volume in order_depth.buy_orders.items():
                print(f"### Buy Oders {volume}x {price}")
            for price, volume in order_depth.sell_orders.items():
                print(f"### Sell Oders {volume}x {price}")
        for product, listing in state.listings.items():
            print(f"# Listing for {product}:")
            print(f"## Symbol: {listing.symbol}")
            print(f"## Denomination: {listing.denomination}")
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
            acceptable_price = 10;  # Participant should calculate this value
            print("## Acceptable price : " + str(acceptable_price))
            print("## Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("## BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("## SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData