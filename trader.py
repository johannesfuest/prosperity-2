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
            "ORCHIDS": 100,
            "CHOCOLATE": 250,
            "STRAWBERRIES": 350,
            "ROSES": 60,
            "GIFT_BASKET": 60,
            "COCONUT": 300,
            "COCONUT_COUPON": 600 
        }
        self.humidity_history = []
        self.sunlight_history = []
        self.orchid_price_history = []
        self.export_tariff_history = []
    
    
    def run(self, state: TradingState):
        result = {}
        traderData = {}
        buy_thresholds = {}
        sell_thresholds = {}
        conversions = None
        for product in ['AMETHYSTS', 'STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = self.get_acceptable_price_for_product(state, product)
            product_vol = get_product_vol(state, product)
            vol_factor_buy = {
                "AMETHYSTS": 0.25,
                "STARFRUIT": 0.5,
            }
            vol_factor_sell = {
                "AMETHYSTS": 0.25,
                "STARFRUIT": 0.5,
            }
            buy_thres = int(np.floor(acceptable_price * (1 - vol_factor_buy[product] * product_vol)))
            sell_thresh = int(np.ceil(acceptable_price * (1 + vol_factor_sell[product] * product_vol)))
            buy_thresholds[product] = buy_thres
            sell_thresholds[product] = sell_thresh
            if len(order_depth.sell_orders) != 0:
                for ask, ask_amount in order_depth.sell_orders.items():
                    if int(ask) <= buy_thresholds[product]:
                        orders.append(Order(product, math.floor(ask), -ask_amount))
            if len(order_depth.buy_orders) != 0:
                for bid, bid_amount in order_depth.buy_orders.items():
                    if int(bid) >= sell_thresholds[product]:
                        orders.append(Order(product, math.ceil(bid), -bid_amount))
            result[product] = orders
            traderData[product] = update_price_history(state, product)
        result = self.adjust_for_position_breaches(result, state, True)
        result = self.adjust_to_exploit_limits(result, state, buy_thresholds, sell_thresholds)
        orders, conversions = self.get_orchid_trades(state)
        result["ORCHIDS"] = orders
        basket_orders, choc_orders, straw_orders, rose_orders = self.get_basket_trades(state)
        result["GIFT_BASKET"] = basket_orders
        result["CHOCOLATE"] = choc_orders
        result["STRAWBERRIES"] = straw_orders
        result["ROSES"] = rose_orders
        #TODO: add coconut and coconut coupon trades
        res_temp = self.get_coconut_trades(state)
        result["COCONUT"] = res_temp["COCONUT"]
        result["COCONUT_COUPON"] = res_temp["COCONUT_COUPON"]
        traderData = json.dumps(traderData)
        return result, conversions, traderData

    def norm_cdf(x):
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    def black_scholes(S, K, T, r, sigma, option_type='call'):
        """
        Calculate European option price using the Black-Scholes formula,
        without using scipy.stats.norm.cdf, using numpy for computation instead.

        Parameters:
            S (float): Current price of the underlying asset.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annual).
            sigma (float): Volatility of the underlying asset (annual).
            option_type (str): Type of option ('call' or 'put').

        Returns:
            float: Price of the option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

        return price
    
    def bisection_method(f, a, b, tol=1e-6, max_iter=1000):
        """
        Find the root of a function using the Bisection method.

        Parameters:
            f (function): The function for which to find the root.
            a (float): Start of the interval.
            b (float): End of the interval.
            tol (float): Tolerance for stopping criterion.
            max_iter (int): Maximum number of iterations.

        Returns:
            float: Approximate root of the function.
        """
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs")

        for n in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            
            if abs(fc) < tol:
                print(f"Found solution after {n} iterations.")
                return c
            elif f(a) * fc < 0:
                b = c
            else:
                a = c

        raise RuntimeError("Exceeded maximum iterations. No solution found.")

    
    def get_implied_coconut_price(option_price, K, T, r, sigma, option_type='call'):
        """
        Calculate the implied price of a coconut (underlying) using the Black-Scholes formula and the current option price.
        We assume fixed implied volatility and solve for the coconut price.

        Parameters:
            option_price (float): Price of the option.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annual).
            sigma (float): Volatility of the underlying asset (annual).
            option_type (str): Type of option ('call' or 'put').

        Returns:
            float: Implied price of a coconut.
        """
        def f(S):
            return black_scholes(S, K, T, r, sigma, option_type) - option_price

        interval = (9000, 11000)
        return bisection_method(f, *interval)
        
        
    
    def get_coconut_trades(self, state):
        thresholds = {
            "COCONUT": 25,
            "COCONUT_COUPON": 15
        }
        option_price = get_mid_price_from_order_book(state.order_depths, "COCONUT_COUPON")
        implied_coconut_price = get_implied_coconut_price(option_price, 10000, 250/365, 0, 0.19226514699995814)
        coconut_mid_price = get_mid_price_from_order_book(state.order_depths, "COCONUT")
        prices = {
            "COCONUT": implied_coconut_price,
            "COCONUT_COUPON": black_scholes(coconut_mid_price, 10000, 250/365, 0, 0.19226514699995814)
        }
        coconut_position = self.get_position("COCONUT", state)
        coconut_coupon_position = self.get_position("COCONUT_COUPON", state)
        
        buy_positions = {
            "COCONUT": coconut_position,
            "COCONUT_COUPON": coconut_coupon_position
        }
        sell_positions = {
            "COCONUT": coconut_position,
            "COCONUT_COUPON": coconut_coupon_position
        }
        print(f'Calced prices: {prices}')
        print(f'Current order book: {state.order_depths["COCONUT"].buy_orders}')
        print(f'{state.order_depths["COCONUT"].sell_orders}')
        print(f'{state.order_depths["COCONUT_COUPON"].buy_orders}')
        print(f'{state.order_depths["COCONUT_COUPON"].sell_orders}')
        result = {}
        for product in ['COCONUT', 'COCONUT_COUPON']:
            orders = []
            #buy the product
            orders_bought = 0
            while True:
                next_best_ask = self.get_best_ask(product, state, orders_bought)
                if next_best_ask[0] is None:
                    break
                else:
                    if next_best_ask[0] < (prices[product] - thresholds[product]):
                        capacity = self.limits[product] - buy_positions[product]
                        if capacity > 0:
                            orders.append(Order(product, next_best_ask[0], min(-next_best_ask[1], capacity)))
                            buy_positions[product] += min(-next_best_ask[1], capacity)
                            if min(-next_best_ask[1], capacity) == -next_best_ask[1]:
                                orders_bought += 1
                        else:
                            break
                    else:
                        break
            #sell the product
            orders_sold = 0
            while True:
                next_best_bid = self.get_best_bid(product, state, orders_sold)
                if next_best_bid[0] is None:
                    break
                else:
                    if next_best_bid[0] > (prices[product] + thresholds[product]):
                        capacity = -self.limits[product] - sell_positions[product]
                        if capacity < 0:
                            orders.append(Order(product, next_best_bid[0], max(-next_best_bid[1], capacity)))
                            sell_positions[product] += max(-next_best_bid[1], capacity)
                            if max(-next_best_bid[1], capacity) == -next_best_bid[1]:
                                orders_sold += 1
                        else:
                            break
                    else:
                        break
            buy_capacity = self.limits[product] - buy_positions[product]
            sell_capacity = -self.limits[product] - sell_positions[product]
            
            if buy_capacity > 0:
                levels = split_number(buy_capacity)
                for i, level in enumerate(levels):
                    # delta = 5 if product == 'COCONUT' else 1
                    orders.append(Order(product, math.floor(prices[product] - thresholds[product] - 1 - i), level))
            if sell_capacity < 0:
                levels = split_number(-sell_capacity)
                # delta = 5 if product == 'COCONUT' else 1
                for i, level in enumerate(levels):
                    orders.append(Order(product, math.ceil(prices[product] + thresholds[product] + 1 + i), -level))
            result[product] = orders
        print(f'Generated orders: {result}')
        return result
                
                
    
    def get_basket_trades(self, state):
        avg_diff = 379.4904833333333
        LIMIT =  55 #adapt this to trade more/less aggressively
        choc_mid = get_mid_price_from_order_book(state.order_depths, "CHOCOLATE")
        straw_mid = get_mid_price_from_order_book(state.order_depths, "STRAWBERRIES")
        rose_mid = get_mid_price_from_order_book(state.order_depths, "ROSES")
        basket_mid = get_mid_price_from_order_book(state.order_depths, "GIFT_BASKET")
        synth_mid = 4 * choc_mid + 6 * straw_mid + rose_mid
        curr_diff = basket_mid - synth_mid
        buy_diff = avg_diff - LIMIT
        sell_diff = avg_diff + LIMIT
        gift_curr_pos = self.get_position("GIFT_BASKET", state)
        choc_curr_pos = self.get_position("CHOCOLATE", state)
        straw_curr_pos = self.get_position("STRAWBERRIES", state)
        rose_curr_pos = self.get_position("ROSES", state)
        choc_orders_ret = []
        straw_orders_ret = []
        rose_orders_ret = []    
        basket_orders_ret = []
        
        if curr_diff < buy_diff:
            buying_gifts = True
            choc_orders_sold = 0
            straw_orders_sold = 0
            rose_orders_sold = 0
            basket_orders_bought = 0
            top_remaining_choc_gone = 0
            top_remaining_straw_gone = 0
            top_remaining_rose_gone = 0
            top_remaining_basket_gone = 0
        
            while buying_gifts:
                #buy baskets and sell components while profitable and within limits 
                possible = True
                if gift_curr_pos + 1 > self.limits["GIFT_BASKET"]:
                    possible = False
                if choc_curr_pos - 4 < -self.limits["CHOCOLATE"]:
                    possible = False
                if straw_curr_pos - 6 < -self.limits["STRAWBERRIES"]:
                    possible = False
                if rose_curr_pos - 1 < -self.limits["ROSES"]:
                    possible = False
                if gift_curr_pos >= 0 and state.timestamp % 1000000 > 950000:
                    possible = False
                if not possible:
                    buying_gifts = False
                    break
                best_basket = self.get_best_ask("GIFT_BASKET", state, basket_orders_bought)
                best_rose = self.get_best_bid("ROSES", state, rose_orders_sold)
                choc_orders, choc_order_sold_temp, top_remaining_choc_gone, total_choc_price, choc_success =\
                    self.get_n_best(state,"CHOCOLATE", False, choc_orders_sold, top_remaining_choc_gone, 4)
                straw_orders, straw_orders_sold_temp, top_remaining_straw_gone, total_straw_price, straw_success =\
                    self.get_n_best(state,"STRAWBERRIES", False, straw_orders_sold, top_remaining_straw_gone, 6)
                if choc_success and straw_success:
                    if (best_rose[0] is None or best_basket[0] is None):
                        buying_gifts = False
                        break
                    synth_price = total_choc_price + total_straw_price + abs(best_rose[0])
                    basket_price = abs(best_basket[0])
                    if basket_price - synth_price < buy_diff:
                        basket_orders_ret.append(Order("GIFT_BASKET", basket_price, 1))
                        choc_orders_ret += choc_orders
                        straw_orders_ret += straw_orders
                        rose_orders_ret.append(Order("ROSES", best_rose[0], -1))
                        gift_curr_pos += 1
                        choc_curr_pos -= 4
                        straw_curr_pos -= 6
                        rose_curr_pos -= 1
                        choc_orders_sold += choc_order_sold_temp
                        straw_orders_sold += straw_orders_sold_temp
                        top_remaining_rose_gone += 1
                        if top_remaining_rose_gone == abs(best_rose[1]):
                            rose_orders_sold += 1
                            top_remaining_rose_gone = 0
                        top_remaining_basket_gone += 1
                        if top_remaining_basket_gone == abs(best_basket[1]):
                            basket_orders_bought += 1
                            top_remaining_basket_gone = 0
                    else:
                        buying_gifts = False
                        # try to market make
                        min_synth_price = basket_price + buy_diff
                        best_rose_price = best_rose[0]
                        best_choc_price = total_choc_price/4
                        best_straw_price = total_straw_price/6
                        rose_ratio = best_rose_price / min_synth_price
                        min_rose_price = min_synth_price - 4 * best_choc_price - 6 * best_straw_price
                        min_choc_price = (min_synth_price - 6 * best_straw_price - best_rose_price) / 4
                        min_straw_price = (min_synth_price - 4 * best_choc_price - best_rose_price) / 6
                        # rose_orders_ret.append(Order("ROSES", min_rose_price+3, -(self.limits["ROSES"] + rose_curr_pos)))
                        # choc_orders_ret.append(Order("CHOCOLATE", min_choc_price+3, -(self.limits["CHOCOLATE"] + choc_curr_pos)))
                        # straw_orders_ret.append(Order("STRAWBERRIES", min_straw_price+3, -(self.limits["STRAWBERRIES"] + straw_curr_pos)))
                        break
                else:
                    buying_gifts = False 
                    break
            return basket_orders_ret, choc_orders_ret, straw_orders_ret, rose_orders_ret
        elif curr_diff > sell_diff:
            selling_gifts = True
            choc_orders_bought = 0
            straw_orders_bought = 0
            rose_orders_bought = 0
            basket_orders_sold = 0
            top_remaining_choc_gone = 0
            top_remaining_straw_gone = 0
            top_remaining_rose_gone = 0
            top_remaining_basket_gone = 0
            while selling_gifts:
                possible = True
                if gift_curr_pos - 1 < -self.limits["GIFT_BASKET"]:
                    possible = False
                if choc_curr_pos + 4 > self.limits["CHOCOLATE"]:
                    possible = False
                if straw_curr_pos + 6 > self.limits["STRAWBERRIES"]:
                    possible = False
                if rose_curr_pos + 1 > self.limits["ROSES"]:
                    possible = False
                #TODO: add end of day stopper to other side too
                if gift_curr_pos <= 0 and state.timestamp % 1000000 > 950000:
                    possible = False
                if not possible:
                    selling_gifts = False
                    break
                best_basket = self.get_best_bid("GIFT_BASKET", state, basket_orders_sold)
    
                best_rose = self.get_best_ask("ROSES", state, rose_orders_bought)
                choc_orders, choc_orders_bought_temp, top_remaining_choc_gone, total_choc_price, choc_success = \
                    self.get_n_best(state, "CHOCOLATE", True, choc_orders_bought, top_remaining_choc_gone, 4)
                straw_orders, straw_orders_bought_temp, top_remaining_straw_gone, total_straw_price, straw_success = \
                    self.get_n_best(state, "STRAWBERRIES", True, straw_orders_bought, top_remaining_straw_gone, 6)
                if (choc_success and straw_success):
                    if (best_rose[0] is None or best_basket[0] is None):
                        selling_gifts = False
                        break
                    synth_price = total_choc_price + total_straw_price + abs(best_rose[0])
                    basket_price = abs(best_basket[0])
                    if basket_price - synth_price > sell_diff:
                        basket_orders_ret.append(Order("GIFT_BASKET", basket_price, -1))
                        choc_orders_ret += choc_orders
                        straw_orders_ret += straw_orders
                        rose_orders_ret.append(Order("ROSES", best_rose[0], 1))
                        gift_curr_pos -= 1
                        choc_curr_pos += 4
                        straw_curr_pos += 6
                        rose_curr_pos += 1
                        choc_orders_bought += choc_orders_bought_temp
                        straw_orders_bought += straw_orders_bought_temp
                        top_remaining_rose_gone += 1
                        if top_remaining_rose_gone == abs(best_rose[1]):
                            rose_orders_bought += 1
                            top_remaining_rose_gone = 0
                        top_remaining_basket_gone += 1
                        if top_remaining_basket_gone == abs(best_basket[1]):
                            basket_orders_sold += 1
                            top_remaining_basket_gone = 0
                    else:
                        selling_gifts = False
                else:
                    selling_gifts = False
            return basket_orders_ret, choc_orders_ret, straw_orders_ret, rose_orders_ret
        else:
            best_basket = self.get_best_bid("GIFT_BASKET", state, 0)
            choc_best_ask, choc_best_ask_amount = self.get_best_ask("CHOCOLATE", state)
            straw_best_ask, straw_best_ask_amount = self.get_best_ask("STRAWBERRIES", state)
            rose_best_ask, rose_best_ask_amount = self.get_best_ask("ROSES", state)

            max_synth_price = abs(best_basket[0]) - sell_diff

            max_rose_price = max_synth_price - 4 * choc_best_ask - 6 * straw_best_ask
            max_choc_price = (max_synth_price - 6 * straw_best_ask - rose_best_ask) / 4
            max_straw_price = (max_synth_price - 4 * choc_best_ask - rose_best_ask) / 6

            # tried market making. but didn't add anything
            # rose_orders_ret.append(Order("ROSES", max_rose_price, 1))
            # choc_orders_ret.append(Order("CHOCOLATE", max_choc_price, 1))
            # straw_orders_ret.append(Order("STRAWBERRIES", max_straw_price, 1))


            
            return basket_orders_ret, choc_orders_ret, straw_orders_ret, rose_orders_ret  

    def get_orchid_trades(self, state):
        
        orchid_data = state.observations.conversionObservations['ORCHIDS']
        bid_price_south = orchid_data.bidPrice
        ask_price_south = orchid_data.askPrice
        mid_price_south = (bid_price_south + ask_price_south) / 2

        # adjust south bid and ask for transport fees
        bid_price_south = bid_price_south - orchid_data.exportTariff - orchid_data.transportFees
        ask_price_south = ask_price_south + orchid_data.importTariff + orchid_data.transportFees
        
        orders = []
        # positive numbers of how many we can buy and sell without hitting the limit
        max_buy_capacity = self.limits['ORCHIDS'] - self.get_position("ORCHIDS", state)
        max_sell_capacity = self.limits['ORCHIDS'] + self.get_position("ORCHIDS", state)

        # get north 
        north_best_bid, north_best_bid_amount = self.get_best_bid("ORCHIDS", state)
        north_best_ask, north_best_ask_amount = self.get_best_ask("ORCHIDS", state)
        # adjust for holding costs
        north_best_ask = north_best_ask + 0.1
        
        expected_profit_dict = {}
        
        # Ship off all previous round orchids
        conversion = self.get_position("ORCHIDS", state)

        # check best 20 orders on either side of book and calculate expected profit of arbitrage trades
        # if arbitrage currently not profitable, don't submit conversion
        #TODO: rather than assuming prices will stay unchanged in the south use next day south price forecast
        for i in range(0, 20):
            bid, bid_amount = self.get_best_bid("ORCHIDS", state, i)
            ask, ask_amount = self.get_best_ask("ORCHIDS", state, i)
            # check whether conversion is profitable. If not use local market instead to close position from previous round
            if bid is not None and ask is not None:
                if bid > bid_price_south:
                    if conversion > 0:
                        conversion_old = conversion
                        conversion = max(conversion - bid_amount, 0)
                        orders.append(Order("ORCHIDS", bid, -max(conversion_old, bid_amount)))
                        max_sell_capacity = max_sell_capacity - abs(-max(conversion_old, bid_amount))
                if ask < ask_price_south:
                    if conversion < 0:
                        conversion_old = conversion
                        conversion = min(conversion + ask_amount, 0)
                        orders.append(Order("ORCHIDS", ask, min(conversion_old, ask_amount)))
                        max_buy_capacity = max_buy_capacity - abs(min(conversion_old, ask_amount))
            #arbitrage for next round         
            if bid is not None:
                # check if can send orchids north
                if (bid > ask_price_south) & (max_sell_capacity > 0):
                    expected_profit_dict[(bid, -bid_amount)] = (bid - ask_price_south) * bid_amount
                    max_sell_capacity = max_sell_capacity - abs(bid_amount)
            if ask is not None:
                #ask = ask + 0.1 -> have to leave this out because ask has to be an int and we already check if its smaller
                # check if can send orchids south
                if (ask < bid_price_south) & (max_buy_capacity > 0):
                    expected_profit_dict[(ask, ask_amount)] = (bid_price_south - ask) * ask_amount
                    max_buy_capacity = max_buy_capacity - abs(ask_amount)
                        
                    
                    
        #TODO: add trades based on domestic price predictions
        def get_last_entry(lst):
            if len(lst) == 0:
                return -1
            return lst[-1]

        buy_orchids = False
        sell_orchids = False
        if len(self.sunlight_history)>0 and (orchid_data.sunlight - self.sunlight_history[-1] > 1):
            buy_orchids = True
        if len(self.sunlight_history)>0 and (orchid_data.sunlight - self.sunlight_history[-1] < -1) and (orchid_data.sunlight - self.sunlight_history[-1] > -2):
            sell_orchids = True
        if len(self.humidity_history)>0 and (orchid_data.humidity - self.humidity_history[-1] < -0.01):
            sell_orchids = True
        if len(self.export_tariff_history)>0 and (orchid_data.exportTariff - self.export_tariff_history[-1] >= 1):
            buy_orchids = True
        if len(self.export_tariff_history)>0 and (orchid_data.exportTariff - self.export_tariff_history[-1] <= -1):
            sell_orchids = True
        if buy_orchids:
            if max_buy_capacity > 0:
                orders.append(Order("ORCHIDS", int(np.floor(bid_price_south-1)), max_buy_capacity))
                max_buy_capacity = 0
        if sell_orchids:
            if max_sell_capacity > 0:
                orders.append(Order("ORCHIDS", int(np.ceil((ask_price_south+1))), -max_sell_capacity))
                max_sell_capacity = 0

        
        
        #fill rest of order capacity with arbitrage trades
        if (max_buy_capacity > 0):
            buy_thresh = bid_price_south
            price_1 = int(np.floor(buy_thresh - 2))
            price_2 = int(np.floor(buy_thresh - 3))
            price_3 = int(np.floor(buy_thresh - 5))
            quantities = split_number(max_buy_capacity)
            expected_profit_dict[(price_1, quantities[0])] = 1
            expected_profit_dict[(price_2, quantities[1])] = 1
            expected_profit_dict[(price_3, quantities[2])] = 1
        if (max_sell_capacity > 0):
            sell_thresh = ask_price_south
            price_1 = int(np.ceil(sell_thresh + 2))
            price_2 = int(np.ceil(sell_thresh + 3))
            price_3 = int(np.ceil(sell_thresh + 5))
            quantities = split_number(max_sell_capacity)
            expected_profit_dict[(price_1, -quantities[0])] = 1
            expected_profit_dict[(price_2, -quantities[1])] = 1
            expected_profit_dict[(price_3, -quantities[2])] = 1

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
        conversion = abs(conversion)
        self.humidity_history.append(orchid_data.humidity)
        self.sunlight_history.append(orchid_data.sunlight)
        self.orchid_price_history.append(mid_price_south)
        self.export_tariff_history.append(orchid_data.exportTariff)
        return orders, conversion
    
    def get_acceptable_price_for_product(self, state, product):

        star_fruit_coefs = np.array([
            #coeffs from round 1 submission
            1.7044926379649041, 0.2920955, 0.20671938, 0.14077617, 0.10025522, 0.08580541 ,0.06038695, 0.03888277, 0.00594952, 0.02262225, 0.01394354, 0.0164973, 0.00535559, 0.00513494, 0.00572899, -0.00049075
            #coeffs using only day 0 to train
            # 13.156199936551275,0.30189398,0.21454386,0.13574109,0.11238089,0.06955258,0.06800676,0.05140635,0.0071232,0.03675125
            ])
        price_history = get_price_history_from_state(state, product)
        
        match product:
            case "STARFRUIT":
                price_history = np.array([1.0] + list(reversed(price_history)))
                predicted_price = np.dot(star_fruit_coefs, price_history)
                return predicted_price
            case _:
                return sum(price_history) / len(price_history)
        
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    
    
    def get_best_bid(self, product, state: TradingState, index=0):
        market_bids = state.order_depths[product].buy_orders
        #best_bid = max(market_bids)
        if len(market_bids) < (index + 1):
            return None, None
        best_bid, best_bid_amount = sorted(list(market_bids.items()), reverse=True)[index]
        return best_bid, best_bid_amount

    def get_best_ask(self, product, state: TradingState, index=0):
        market_asks = state.order_depths[product].sell_orders
        #best_ask = min(market_asks)
        if len(market_asks) < (index + 1):
            return None, None
        best_ask, best_ask_amount = sorted(list(market_asks.items()))[index]
        return best_ask, best_ask_amount 
    
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
            
    def get_n_best(self, state, product, buy, orders_bought, top_order_already_gone, n):
        n_product_found = 0
        success = False
        order_list = []
        orders_cleared = 0
        best_order = None
        while n_product_found < n:
            #get best available remaining order on book
            if buy:
                best_order = self.get_best_ask(product, state, orders_bought)
            else:
                best_order = self.get_best_bid(product, state, orders_bought)
            if best_order[0] is None:
                break
            #see how much quantity is left for us in the best remaining order
            found_this_round = min(n - n_product_found, abs(best_order[1]) - top_order_already_gone)
            top_order_already_gone += found_this_round
            n_product_found += found_this_round
            #check if we used up the whole order and need to move to the next one
            if abs(best_order[1]) == top_order_already_gone:
                orders_cleared += 1
                top_order_already_gone = 0
            # append the order
            if buy:
                order_list.append(Order(product, best_order[0], found_this_round))
            else:
                order_list.append(Order(product, best_order[0], -found_this_round))
            # check if we have found n products and are done
            if n_product_found == n:
                success = True
                break
        if success:
            total_price = sum([abs(order.price * abs(order.quantity)) for order in order_list])
        else:
            total_price = 9999999
        return order_list, orders_cleared, top_order_already_gone, total_price, success
    
    
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
    if len(order_depth[product].buy_orders) == 0:
        return list(order_depth[product].sell_orders.keys())[0]
    if len(order_depth[product].sell_orders) == 0:
        return list(order_depth[product].buy_orders.keys())[0]
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

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate European option price using the Black-Scholes formula,
    without using scipy.stats.norm.cdf, using numpy for computation instead.

    Parameters:
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying asset (annual).
        option_type (str): Type of option ('call' or 'put').

    Returns:
        float: Price of the option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return price

def bisection_method(f, a, b, tol=1e-7, max_iter=2000):
    """
    Find the root of a function using the Bisection method.

    Parameters:
        f (function): The function for which to find the root.
        a (float): Start of the interval.
        b (float): End of the interval.
        tol (float): Tolerance for stopping criterion.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: Approximate root of the function.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    for n in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tol:
            print(f"Found solution after {n} iterations.")
            return c
        elif f(a) * fc < 0:
            b = c
        else:
            a = c

    raise RuntimeError("Exceeded maximum iterations. No solution found.")


def get_implied_coconut_price(option_price, K, T, r, sigma, option_type='call'):
    """
    Calculate the implied price of a coconut (underlying) using the Black-Scholes formula and the current option price.
    We assume fixed implied volatility and solve for the coconut price.

    Parameters:
        option_price (float): Price of the option.
        K (float): Strike price of the option.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying asset (annual).
        option_type (str): Type of option ('call' or 'put').

    Returns:
        float: Implied price of a coconut.
    """
    def f(S):
        return black_scholes(S, K, T, r, sigma, option_type) - option_price

    interval = (9500, 10500)
    return bisection_method(f, *interval)
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