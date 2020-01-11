from tensorforce.agents import Agent
from tensorforce.environments import Environment

from tensortrade.environments import TradingEnvironment
# from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.exchanges.simulated.fbm_exchange import FBMExchange
# from tensortrade.exchanges.simulated import simulated_exchange as FBMExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfit
from tensortrade.actions.discrete_actions import DiscreteActions
# from tensortrade.actions import DynamicOrders as DiscreteActions
from tensortrade.strategies import TensorforceTradingStrategy

normalize = MinMaxNormalizer(inplace=True)
difference = FractionalDifference(difference_order=0.6,
                                  inplace=True)
feature_pipeline = FeaturePipeline(steps=[normalize, difference])

reward_scheme = SimpleProfit()
action_scheme = DiscreteActions(n_actions=20, instrument='ETH/BTC')

exchange = FBMExchange(base_instrument='BTC',
                       timeframe='1h',
                       should_pretransform_obs=True)

environment = TradingEnvironment(exchange=exchange,
                                 action_scheme=action_scheme,
                                 reward_scheme=reward_scheme,
                                 feature_pipeline=feature_pipeline)

agent_spec = {
    "type": "ppo_agent",
    "step_optimizer": {
        "type": "adam",
        "learning_rate": 1e-4
    },
    "discount": 0.99,
    "likelihood_ratio_clipping": 0.2,
}

network_spec = [
    dict(type='dense', size=64, activation="tanh"),
    dict(type='dense', size=32, activation="tanh")
]

strategy = TensorforceTradingStrategy(environment=environment, agent_spec=agent_spec, network_spec=network_spec)

performance = strategy.run(episodes=1, testing=True)

print(performance[-5:])