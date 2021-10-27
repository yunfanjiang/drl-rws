from ray.rllib.models import ModelCatalog

from .baseline import BaselineModel


ModelCatalog.register_custom_model("baseline_model", BaselineModel)
