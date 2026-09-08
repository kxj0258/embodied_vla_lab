import numpy as np

from evla_benchmark.evaluator import EvaluatorConfig, evaluate, validate_action
from evla_benchmark.schema import ActionSpec, ObservationSpec, TaskSpec
from evla_benchmark.testing import MockEnvironment, MockPolicy


def task(states=(0, 1)):
    return TaskSpec(
        suite="mock", task_id=0, name="success", instruction="", split="development",
        observation=ObservationSpec(state_dim=1), action=ActionSpec(dim=2), episode_horizon=5,
        init_state_indices=tuple(states),
    )


def test_validate_action_rejects_nonfinite_and_wrong_shape():
    try:
        validate_action([0.0], 2, -1.0, 1.0)
        raise AssertionError("wrong shape accepted")
    except ValueError:
        pass
    try:
        validate_action([np.nan, 0.0], 2, -1.0, 1.0)
        raise AssertionError("NaN accepted")
    except ValueError:
        pass


def test_mock_runtime_evaluator():
    env = MockEnvironment(success_after=2)
    result = evaluate(MockPolicy(), env, task(), EvaluatorConfig("test", "mock", 10, 5))
    assert result.success_count == 2
    assert result.success_total == 2
    assert result.failures == {"success": 2}
    assert env.closed
