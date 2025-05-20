import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import uuid
import pickle
import numpy as np
from datetime import datetime
from utils import Utils

class PortfolioTracker:
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}  # {ticker: {quantity, avg_price}}
        self.history = []
        self.transactions = []
        self.current_position_value = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.total_fees = 0  # Track total fees paid
        # Buy-and-hold portfolio
        self.bh_holdings = {}  # {ticker: {quantity, buy_price}}
        self.bh_history = []
        self.bh_position_value = 0
        
        # Initialize history with starting point
        self._add_history_point(None, None)


    def buy(self, ticker, quantity, price, fee_percentage=0.001, timestamp=None):
        cost = quantity * price
        cost = float(cost)  # Convert to Python float if needed
        
        # Calculate fee
        fee_amount = cost * fee_percentage
        total_cost = cost + fee_amount
        
        if total_cost > self.cash:
            return False, "Insufficient funds"
        
        transaction_id = str(uuid.uuid4())[:8]
        
        # Update holdings
        if ticker in self.holdings:
            # Calculate new average price
            total_shares = self.holdings[ticker]['quantity'] + quantity
            total_cost_basis = (self.holdings[ticker]['quantity'] * self.holdings[ticker]['avg_price']) + cost
            self.holdings[ticker]['avg_price'] = total_cost_basis / total_shares
            self.holdings[ticker]['quantity'] = total_shares
        else:
            self.holdings[ticker] = {
                'quantity': quantity,
                'avg_price': price
            }
        
        # Update cash, position value, and fees
        self.cash -= total_cost
        self.current_position_value += cost
        self.total_fees += fee_amount
        
        # Record transaction
        self.transactions.append({
            'id': transaction_id,
            'timestamp': timestamp if timestamp is not None else datetime.now(),
            'ticker': ticker,
            'action': 'Buy',
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'fee': fee_amount,
            'total_cost': total_cost,
            'fee_percentage': fee_percentage * 100,
            'remaining_cash': self.cash
        })
        
        # Update history
        self._add_history_point('Buy', price, timestamp)
        
        return True, transaction_id

    def sell(self, ticker, quantity, price, fee_percentage=0.001, timestamp=None):
        if ticker not in self.holdings or self.holdings[ticker]['quantity'] < quantity:
            return False, "Insufficient holdings"
        
        transaction_id = str(uuid.uuid4())[:8]
        gross_revenue = quantity * price
        
        # Calculate fee
        fee_amount = gross_revenue * fee_percentage
        net_revenue = gross_revenue - fee_amount
        
        # Calculate realized profit/loss for this transaction
        avg_price = self.holdings[ticker]['avg_price']
        transaction_pnl = (price - avg_price) * quantity - fee_amount
        
        # Update holdings
        self.holdings[ticker]['quantity'] -= quantity
        if self.holdings[ticker]['quantity'] == 0:
            del self.holdings[ticker]
        
        # Update cash, position value, fees, and P&L
        self.cash += net_revenue
        self.current_position_value -= (avg_price * quantity)
        self.realized_pnl += transaction_pnl
        self.total_fees += fee_amount
        
        # Record transaction
        self.transactions.append({
            'id': transaction_id,
            'timestamp': timestamp if timestamp is not None else datetime.now(),
            'ticker': ticker,
            'action': 'Sell',
            'quantity': quantity,
            'price': price,
            'gross_revenue': gross_revenue,
            'fee': fee_amount,
            'net_revenue': net_revenue,
            'fee_percentage': fee_percentage * 100,
            'pnl': transaction_pnl,
            'remaining_cash': self.cash
        })
        
        # Update history
        self._add_history_point('Sell', price, timestamp)
        
        return True, transaction_id


    def _add_history_point(self, action, ticker_price, timestamp=None):
        timestamp = timestamp if timestamp is not None else datetime.now()
        total_value = self.cash + self.current_position_value
        
        self.history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': self.current_position_value,
            'total_value': total_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'ticker_price': ticker_price,
            'total_fees': self.total_fees,
            'action': action
        })
        Utils.log_message(f"DEBUG: Added history point: action={action}, ticker_price={ticker_price}, total_value={total_value:.2f}, timestamp={timestamp}, history_length={len(self.history)}")

    def update_position_values(self, ticker_prices, timestamp=None):
        """Update position values with current market prices"""
        total_position_value = 0
        self.unrealized_pnl = 0
        
        for ticker, details in self.holdings.items():
            if ticker in ticker_prices:
                current_price = ticker_prices[ticker]
                position_value = details['quantity'] * current_price
                position_pnl = (current_price - details['avg_price']) * details['quantity']
                self._add_history_point(None, current_price, timestamp)
                
                total_position_value += position_value
                self.unrealized_pnl += position_pnl
        
        self.current_position_value = total_position_value

    def save_history(self, model_dir, ticker):
        try:
            os.makedirs(model_dir, exist_ok=True)
            history_path = os.path.join(model_dir, f"{ticker}_portfolio_history.pkl")
            with open(history_path, "wb") as f:
                pickle.dump(self.history, f)
            bh_history_path = os.path.join(model_dir, f"{ticker}_bh_portfolio_history.pkl")
            with open(bh_history_path, "wb") as f:
                pickle.dump(self.bh_history, f)
            Utils.log_message(f"INFO: Saved portfolio history to {history_path} and buy-and-hold history to {bh_history_path}")
        except Exception as e:
            Utils.log_message(f"ERROR: Failed to save portfolio history: {e}")
            raise
    
    def reset(self):
        """Reset the portfolio to its initial state."""
        self.cash = self.initial_cash
        self.holdings = {}
        self.history = []
        self.transactions = []
        self.current_position_value = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.total_fees = 0
        self.bh_holdings = {}
        self.bh_history = []
        self.bh_position_value = 0
        self._add_history_point(None, None)


    def init_buy_and_hold(self, ticker, price, fee_percentage=0.001, timestamp=None):
        cost = self.initial_cash / (1 + fee_percentage)
        quantity = int(cost / price)
        fee_amount = quantity * price * fee_percentage
        self.bh_holdings[ticker] = {
            'quantity': quantity,
            'buy_price': price
        }
        self.bh_position_value = quantity * price
        self._add_bh_history_point(ticker, price, timestamp)
        Utils.log_message(f"INFO: Initialized B&H portfolio: {quantity} shares of {ticker} at €{price:.2f}, fee: €{fee_amount:.2f}, position_value: €{self.bh_position_value:.2f}")

    def _add_bh_history_point(self, ticker, price, timestamp=None):
        timestamp = timestamp if timestamp is not None else datetime.now()
        total_value = self.bh_position_value
        cash = self.initial_cash - self.bh_position_value if self.bh_holdings else self.initial_cash
        
        self.bh_history.append({
            'timestamp': timestamp,
            'cash': cash,
            'position_value': self.bh_position_value,
            'total_value': total_value,
            'ticker': ticker,
            'price': price
        })
        Utils.log_message(f"DEBUG: Added B&H history point: ticker={ticker}, price={price}, cash={cash:.2f}, position_value={self.bh_position_value:.2f}, total_value={total_value:.2f}, timestamp={timestamp}")

    def update_bh_position_value(self, ticker, price, timestamp=None):
        updated = False
        for t, details in self.bh_holdings.items():
            if t == ticker:
                self.bh_position_value = details['quantity'] * price
                updated = True
                Utils.log_message(f"DEBUG: Updated B&H position: {ticker}, quantity: {details['quantity']}, price: €{price:.2f}, position_value: €{self.bh_position_value:.2f}, timestamp: {timestamp}")
                self._add_bh_history_point(ticker, price, timestamp)
        if not updated:
            Utils.log_message(f"WARNING: Skipped B&H history update: ticker {ticker} not found in bh_holdings")
