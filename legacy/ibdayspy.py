from ib_insync import *

ib = IB()
ib.connect("127.0.0.1", 4001, clientId=2, timeout=10)

# 3 = delayed market data
ib.reqMarketDataType(3)

contract = Stock("SPY", "SMART", "USD")
ib.qualifyContracts(contract)

ticker = ib.reqMktData(contract, "", False, False)
ib.sleep(5)

print("Bid:", ticker.bid)
print("Ask:", ticker.ask)
print("Last:", ticker.last)
print("Close:", ticker.close)

ib.disconnect()