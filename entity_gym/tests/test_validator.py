import numpy as np
from ragged_buffer import RaggedBufferI64

from entity_gym.env.env_list import EnvList
from entity_gym.env.validator import ValidatingEnv
from entity_gym.examples.minesweeper import MineSweeper


def test_env_list() -> None:
    # 100 environments
    envs = EnvList(lambda: ValidatingEnv(MineSweeper()), 100)
    obs_space = envs.obs_space()

    obs_reset = envs.reset(obs_space)
    assert len(obs_reset.done) == 100

    actions = {
        "Move": RaggedBufferI64.from_array(np.zeros((100, 2, 1), np.int64)),
        "Fire Orbital Cannon": RaggedBufferI64.from_array(
            np.zeros((100, 0, 1), np.int64)
        ),
    }
    obs_act = envs.act(actions, obs_space)

    assert len(obs_act.done) == 100
