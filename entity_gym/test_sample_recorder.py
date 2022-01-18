import numpy as np
import tempfile
from entity_gym.sample_recorder import Sample, SampleRecorder, Trace
from entity_gym.environment import CategoricalActionMaskBatch, ObsBatch
from numpy.lib.twodim_base import mask_indices
from ragged_buffer import RaggedBufferF32, RaggedBufferI64


def test_serde_sample() -> None:
    sample = Sample(
        obs=ObsBatch(
            entities={
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
            action_masks={
                "move": CategoricalActionMaskBatch(
                    actors=RaggedBufferI64.from_array(np.array([[[0]]])), masks=None
                ),
                "shoot": CategoricalActionMaskBatch(
                    actors=RaggedBufferI64.from_array(np.array([[[0]]])), masks=None
                ),
                "explode": CategoricalActionMaskBatch(
                    actors=RaggedBufferI64.from_array(np.array([[[4], [5], [6]]])),
                    masks=None,
                ),
            },
            reward=np.array([0.3124125987123489]),
            ids=[[0, 1, 2, 3, 4, 5, 6]],
            done=np.array([False]),
            end_of_episode_info={},
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
        actions=[],
        step=[13],
        episode=[4213],
    )

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        sample_recorder = SampleRecorder(f.name, act_space=None, obs_space=None)  # type: ignore
        sample_recorder.record(sample)
        # modify the sample
        sample.obs.reward = np.array([1.0])
        sample.obs.entities["hero"] = RaggedBufferF32.from_array(
            np.array([[[1.0, 2.0, 0.3, 200.0, 10.0]]], dtype=np.float32),
        )
        sample_recorder.record(sample)
        sample_recorder.close()

        with open(f.name, "rb") as f:
            trace = Trace.deserialize(f.read())
            assert len(trace.samples) == 2
            assert trace.samples[0].obs.reward[0] == 0.3124125987123489
            assert trace.samples[1].obs.reward[0] == 1.0
            np.testing.assert_equal(
                trace.samples[0].obs.entities["hero"][0].as_array(),
                np.array([[1.0, 2.0, 0.3, 100.0, 10.0]], dtype=np.float32),
            )
            np.testing.assert_equal(
                trace.samples[1].obs.entities["hero"][0].as_array(),
                np.array([[1.0, 2.0, 0.3, 200.0, 10.0]], dtype=np.float32),
            )
