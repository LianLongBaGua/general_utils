from elite_ctastrategy import (
    rsi,
    boll,
    std,
    cross_over,
    cross_below,
    atr,
    natr, 
    HistoryManager
)

from numpy import ndarray

class CommonFactors:
    def calculate_moving_stoploss(self, last_target: int, hm: HistoryManager, natr_risk_window: int=20, natr_multiplier: float=4):
        """
        Use the common natr function to calculate the moving stoploss
        natr_risk_window: int = 20
        natr_multiplier: float = 4
        """
        self.atr_risk_value = atr(hm.high, hm.low, hm.close, self.atr_risk_window)[-1]

        if last_target == 0:
            self.intra_trade_high = hm.high[-1]
            self.intra_trade_low = hm.low[-1]
            return last_target
        if last_target > 0:
            self.intra_trade_high = max(self.intra_trade_high, hm.high[-1])
            moving_stop_price: float = (
                self.intra_trade_high - self.atr_risk_value * self.atr_multiplier
            )
            if hm.close[-1] <= moving_stop_price:
                return 0
        elif last_target < 0:
            self.intra_trade_low = min(self.intra_trade_low, hm.low[-1])
            moving_stop_price: float = (
                self.intra_trade_low + self.atr_risk_value * self.atr_multiplier
            )
            if hm.close[-1] >= moving_stop_price:
                return 0
    
    def rsi_filter(self, hm: HistoryManager, rsi_window: int, rsi_upper: float=70, rsi_lower: float=30):
        """
        Just to make sure we are not entering when it is overbought or oversold
        rsi_upper: float = 70
        rsi_lower: float = 30
        """
        rsi_array: ndarray = rsi(hm.close, rsi_window)
        rsi_filter: bool = rsi_lower < rsi_array[-1] < rsi_mid
        return rsi_filter