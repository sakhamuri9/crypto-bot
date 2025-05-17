"""Simple grid trading bot."""
import logging
import time
import argparse
from binance_client import BinanceClient
import config

logger = logging.getLogger(__name__)

class GridBot:
    """Grid trading strategy that places limit orders at predefined levels."""

    def __init__(self, client, symbol, lower_price, upper_price, levels, order_size):
        """Initialize the grid bot parameters."""
        self.client = client
        self.symbol = symbol
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.levels = levels
        self.order_size = order_size

        # Grid spacing derived from config or calculated from range
        self.grid_spacing = getattr(config, "GRID_SPACING", None)
        if self.grid_spacing is None:
            self.grid_spacing = (self.upper_price - self.lower_price) / max(self.levels - 1, 1)

        self.max_open_orders = getattr(config, "MAX_GRID_ORDERS", self.levels)
        self.active_orders = {}

    def _place_limit_order(self, side, price):
        """Place a limit order via Binance API."""
        try:
            order = self.client.client.create_order(
                symbol=self.symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=self.order_size,
                price=f"{price:.8f}"
            )
            logger.info(f"Placed {side} order at {price}")
            return order['orderId']
        except Exception as e:
            logger.error(f"Error placing order at {price}: {e}")
            return None

    def _place_and_track(self, side, price):
        if len(self.active_orders) >= self.max_open_orders:
            return
        if price < self.lower_price or price > self.upper_price:
            return
        order_id = self._place_limit_order(side, price)
        if order_id:
            self.active_orders[price] = (side, order_id)

    def setup_grid(self, current_price):
        """Place initial grid orders around the current price."""
        levels = [self.lower_price + i * self.grid_spacing for i in range(self.levels)]
        for level in levels:
            if len(self.active_orders) >= self.max_open_orders:
                break
            if level < current_price:
                self._place_and_track('BUY', level)
            elif level > current_price:
                self._place_and_track('SELL', level)

    def check_orders(self):
        """Check existing orders and reposition when filled."""
        for price, (side, order_id) in list(self.active_orders.items()):
            try:
                order = self.client.client.get_order(symbol=self.symbol, orderId=order_id)
            except Exception as e:
                logger.error(f"Failed to fetch order status: {e}")
                continue
            status = order.get('status')
            if status == 'FILLED':
                del self.active_orders[price]
                if side == 'BUY':
                    new_price = price + self.grid_spacing
                    self._place_and_track('SELL', new_price)
                else:
                    new_price = price - self.grid_spacing
                    self._place_and_track('BUY', new_price)
            elif status in ('CANCELED', 'REJECTED', 'EXPIRED'):
                del self.active_orders[price]

    def run(self, poll_interval=30):
        """Main loop for the grid bot."""
        ticker = self.client.client.get_symbol_ticker(symbol=self.symbol)
        current_price = float(ticker['price'])
        logger.info(f"Starting grid bot at price {current_price}")
        self.setup_grid(current_price)
        while True:
            self.check_orders()
            time.sleep(poll_interval)

def run_grid_bot():
    """Command line entry point to run the grid bot."""
    parser = argparse.ArgumentParser(description='Grid Trading Bot')
    parser.add_argument('--symbol', type=str, default=config.SYMBOL,
                        help=f'Trading pair symbol (default: {config.SYMBOL})')
    parser.add_argument('--lower-price', type=float, required=True,
                        help='Lower bound of grid price range')
    parser.add_argument('--upper-price', type=float, required=True,
                        help='Upper bound of grid price range')
    parser.add_argument('--levels', type=int, default=5,
                        help='Number of grid levels')
    parser.add_argument('--order-size', type=float, default=0.001,
                        help='Order size at each grid level')
    parser.add_argument('--poll-interval', type=int, default=30,
                        help='Polling interval in seconds')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    client = BinanceClient(testnet=True)
    bot = GridBot(client, args.symbol, args.lower_price, args.upper_price,
                  args.levels, args.order_size)
    bot.run(poll_interval=args.poll_interval)

if __name__ == '__main__':
    run_grid_bot()
