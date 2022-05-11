import tempfile

import numpy as np
from ragged_buffer import RaggedBufferBool, RaggedBufferF32, RaggedBufferI64

from entity_gym.env import VecCategoricalActionMask, VecObs
from entity_gym.serialization import Sample, SampleRecorder, Trace


def test_serde_sample() -> None:
    sample = Sample(
        obs=VecObs(
            features={
                "hero": RaggedBufferF32.from_array(
                    np.array([[[1.0, 2.0, 0.3, 100.0, 10.0]]], dtype=np.float32),
                ),
                "enemy": RaggedBufferF32.from_array(
                    np.array(
                        [
                            [
                                [4.0, -2.0, 0.3, 100.0],
                                [5.0, -2.0, 0.3, 100.0],
                                [6.0, -2.0, 0.3, 100.0],
                            ]
                        ],
                        dtype=np.float32,
                    ),
                ),
                "box": RaggedBufferF32.from_array(
                    np.array(
                        [
                            [
                                [0.0, 0.0, 0.3, 100.0],
                                [1.0, 0.0, 0.3, 100.0],
                                [2.0, 0.0, 0.3, 100.0],
                            ]
                        ],
                        dtype=np.float32,
                    ),
                ),
            },
            visible={},
            action_masks={
                "move": VecCategoricalActionMask(
                    actors=RaggedBufferI64.from_array(np.array([[[0]]])),
                    mask=RaggedBufferBool.from_array(np.array([[[True, False, True]]])),
                ),
                "shoot": VecCategoricalActionMask(
                    actors=RaggedBufferI64.from_array(np.array([[[0]]])), mask=None
                ),
                "explode": VecCategoricalActionMask(
                    actors=RaggedBufferI64.from_array(np.array([[[4], [5], [6]]])),
                    mask=None,
                ),
            },
            reward=np.array([0.3124125987123489]),
            done=np.array([False]),
            metrics={},
        ),
        probs={
            "move": RaggedBufferF32.from_array(
                np.array([[[0.5], [0.2], [0.3], [0.0]]], dtype=np.float32)
            ),
            "shoot": RaggedBufferF32.from_array(
                np.array([[[0.9], [0.1]]], dtype=np.float32)
            ),
            "explode": RaggedBufferF32.from_array(
                np.array(
                    [[[0.3], [0.7]], [[0.2], [0.8]], [[0.1], [0.9]]], dtype=np.float32
                )
            ),
        },
        logits=None,
        actions={},
        step=[13],
        episode=[4213],
    )

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        sample_recorder = SampleRecorder(f.name, act_space=None, obs_space=None, subsample=1)  # type: ignore
        sample_recorder.record(sample)
        # modify the sample
        sample.obs.reward = np.array([1.0])
        sample.obs.features["hero"] = RaggedBufferF32.from_array(
            np.array([[[1.0, 2.0, 0.3, 200.0, 10.0]]], dtype=np.float32),
        )
        sample_recorder.record(sample)
        sample_recorder.close()

        with open(f.name, "rb") as f2:
            trace = Trace.deserialize(f2.read())
            assert len(trace.samples) == 2
            assert trace.samples[0].obs.reward[0] == 0.3124125987123489
            assert trace.samples[1].obs.reward[0] == 1.0
            assert (
                trace.samples[0].obs.action_masks["move"]
                == sample.obs.action_masks["move"]
            )
            np.testing.assert_equal(
                trace.samples[0].obs.features["hero"][0].as_array(),
                np.array([[1.0, 2.0, 0.3, 100.0, 10.0]], dtype=np.float32),
            )
            np.testing.assert_equal(
                trace.samples[1].obs.features["hero"][0].as_array(),
                np.array([[1.0, 2.0, 0.3, 200.0, 10.0]], dtype=np.float32),
            )
