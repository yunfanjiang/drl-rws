from typing import Optional

import dm_env
from dm_env import specs
from gym import spaces


def spec2space(spec: dm_env.specs, name: Optional[str] = None):
    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(n=spec.num_values)
    elif isinstance(spec, specs.BoundedArray):
        if (
            spec.minimum == 0.0
            and spec.minimum == 1.0
            and spec.dtype in ("int32", "int64")
        ):
            return spaces.MultiBinary(n=spec.shape)
        elif spec.minimum == 0.0 and spec.dtype in ("int32", "int64"):
            return spaces.MultiDiscrete(nvec=spec.maximum, dtype=spec.dtype)
        else:
            return spaces.Box(
                low=float(spec.minimum),
                high=float(spec.maximum),
                shape=spec.shape,
                dtype=spec.dtype,
            )
    elif isinstance(spec, specs.Array):
        if spec.dtype == "int32":
            # observation is index of piece, max = 41, min = 0
            return spaces.Box(shape=spec.shape, dtype=spec.dtype, low=0, high=41,)
        elif spec.dtype == "uint8":
            # image observation
            return spaces.Box(shape=spec.shape, dtype=spec.dtype, low=0, high=255)
        else:
            raise ValueError
    elif isinstance(spec, tuple):
        return spaces.Tuple(tuple(spec2space(s, name) for s in spec))
    elif isinstance(spec, dict):
        return spaces.Dict(
            {key: spec2space(value, name) for key, value in spec.items()}
        )
    else:
        raise ValueError("Unexpected dmlab2d space: {}".format(spec))
