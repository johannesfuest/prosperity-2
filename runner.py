import time

from datamodel import Listing, OrderDepth, Trade, TradingState
from trader2 import Trader

timestamp = time.time()

listings = {
	"AMETHYSTS": Listing(
		symbol="AMETHYSTS", 
		product="AMETHYSTS", 
		denomination= "SEASHELLS"
	),
	"STARFRUIT": Listing(
		symbol="STARFRUIT", 
		product="STARFRUIT", 
		denomination= "SEASHELLS"
	),
    "ORCHIDS": Listing(
		symbol="ORCHIDS", 
		product="ORCHIDS", 
		denomination= "SEASHELLS"
	),
}

order_depths = {
	"AMETHYSTS": OrderDepth(
		buy_orders={10: 7, 9: 5},
		sell_orders={11: -4, 12: -8}
	),
	"STARFRUIT": OrderDepth(
		buy_orders={142: 3, 141: 5},
		sell_orders={144: -5, 145: -8}
	),
    "ORCHIDS": OrderDepth(
		buy_orders={1200: 3, 1300: 5},
		sell_orders={1500: -5, 1240: -8}
	),	
}

own_trades = {
	"AMETHYSTS": [],
	"STARFRUIT": [],
    "ORCHIDS": []
}

market_trades = {
	"AMETHYSTS": [
		Trade(
			symbol="AMETHYSTS",
			price=11,
			quantity=4,
			buyer="",
			seller="",
			timestamp=900
		)
	],
	"STARFRUIT": [],
    "ORCHIDS": []
}

position = {
	"AMETHYSTS": 3,
	"STARFRUIT": -5,
    "ORCHIDS": 0
}

observations = {}
traderData = ""

state = TradingState(
	traderData,
	timestamp,
    listings,
	order_depths,
	own_trades,
	market_trades,
	position,
	observations
)

trader = Trader()
result, conversions, traderData = trader.run(state)

print(f"Result: {result}")
print(f"Conversions: {conversions}")
print(f"TraderData: {traderData}")
