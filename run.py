import tensorflow as tf
import pandas as pd

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2

from tensortrade.strategies import StableBaselinesTradingStrategy
from tensortrade.environments import TradingEnvironment
from tensortrade.rewards import RiskAdjustedReturns
from tensortrade.actions import ManagedRiskOrders
from tensortrade.instruments import Quantity, TradingPair, BTC, USD
# from tensortrade.orders.criteria import StopLoss, StopDirection
from tensortrade.orders.criteria import StopDirection
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.features.stationarity import LogDifference
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features import FeaturePipeline

WINDOW_SIZE = 2
PRICE_COLUMN = 'close'

normalize = MinMaxNormalizer(inplace=True)
difference = LogDifference(inplace=True)
feature_pipeline = FeaturePipeline(steps=[normalize])

action_scheme = ManagedRiskOrders(pairs=[USD/BTC])
reward_scheme = RiskAdjustedReturns(return_algorithm="sortino")

# csv_file = tf.keras.utils.get_file(
    # 'Coinbase_BTCUSD_1h.csv', 'https://www.cryptodatadownload.com/cdd/Coinbase_BTCUSD_1h.csv')
ohlcv_data = pd.read_csv('./examples/data/Coinbase_BTCUSD_1h.csv', skiprows=1)
# ohlcv_data = pd.read_csv(csv_file, skiprows=1, index_col="Date")
ohlcv_data.columns = map(str.lower, ohlcv_data.columns)
ohlcv_data = ohlcv_data.rename(columns={'volume btc': 'volume'})

exchange = SimulatedExchange(data_frame=ohlcv_data, price_column=PRICE_COLUMN, randomize_time_slices=True)

wallets = [(exchange, USD, 10000), (exchange, BTC, 0)]

portfolio = Portfolio(base_instrument=USD, wallets=wallets)

environment = TradingEnvironment(exchange=exchange,
                                 portfolio=portfolio,
                                 action_scheme=action_scheme,
                                 reward_scheme=reward_scheme,
                                 feature_pipeline=feature_pipeline,
                                 window_size=WINDOW_SIZE,
                                 observe_wallets=[USD, BTC])

print('Observation Data:')
print(environment.observation_columns)

model = PPO2
policy = MlpLnLstmPolicy
params = { "learning_rate": 1e-5, 'nminibatches': 1 }

strategy = StableBaselinesTradingStrategy(environment=environment,
                                          model=model,
                                          policy=policy,
                                          model_kwargs=params)

strategy.run(steps=10000)

strategy.save_agent(path="agents/test")