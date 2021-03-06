from ray.rllib.models import ModelCatalog

from .baseline import BaselineModel
from .centralized_critic import CCFFModel, CCRNNModel
from .dqn_model import DQNTorchModel

ModelCatalog.register_custom_model("baseline_model", BaselineModel)
ModelCatalog.register_custom_model("centralized_critic_feed_forward_model", CCFFModel)
ModelCatalog.register_custom_model("centralized_critic_recurrent_model", CCRNNModel)
ModelCatalog.register_custom_model("dqn_model",DQNTorchModel)