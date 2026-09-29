"""Input-gradient regression tests using independent central differences."""
import numpy as np
import pytest
from mlfromscratch.deep_learning.layers import RNN

class RecordGradient:

    def update(self, weights, gradient):
        self.gradient = gradient.copy()
        return weights

def make_layer(seed=0, steps=4, truncation=10, units=3):
    rng = np.random.default_rng(seed)
    layer = RNN(units, activation='tanh', bptt_trunc=truncation, input_shape=(steps, 2))
    layer.U = rng.normal(0, 0.3, (units, 2))
    layer.V = rng.normal(0, 0.3, (2, units))
    layer.W = rng.normal(0, 0.3, (units, units))
    layer.U_opt, layer.V_opt, layer.W_opt = (RecordGradient() for _ in range(3))
    return (layer, rng.normal(0, 0.3, (2, steps, 2)), rng.normal(0, 0.5, (2, steps, 2)))

def forward_reference(layer, x):
    state = np.zeros((x.shape[0], layer.n_units))
    outputs = []
    for t in range(x.shape[1]):
        state = np.tanh(x[:, t] @ layer.U.T + state @ layer.W.T)
        outputs.append(state @ layer.V.T)
    return np.stack(outputs, axis=1)

def differences(x, objective):
    result = np.zeros_like(x)
    step = 1e-06
    for index in np.ndindex(x.shape):
        positive, negative = (x.copy(), x.copy())
        positive[index] += step
        negative[index] -= step
        result[index] = (objective(positive) - objective(negative)) / (2 * step)
    return result

@pytest.mark.parametrize('seed', [0, 7, 19])
@pytest.mark.parametrize('steps', [2, 4])
def test_full_bptt_matches_input_finite_differences(seed, steps):
    layer, x, upstream = make_layer(seed, steps)
    np.testing.assert_allclose(layer.forward_pass(x), forward_reference(layer, x), rtol=1e-12, atol=1e-12)
    actual = layer.backward_pass(upstream)
    expected = differences(x, lambda candidate: np.sum(forward_reference(layer, candidate) * upstream))
    np.testing.assert_allclose(actual, expected, rtol=2e-05, atol=1e-09)

def test_final_output_supplies_gradient_to_earlier_inputs():
    layer, x, upstream = make_layer(11)
    upstream[:, :-1] = 0
    layer.forward_pass(x)
    gradient = layer.backward_pass(upstream)
    assert np.linalg.norm(gradient[:, :-1]) > 1e-06
    expected = differences(x, lambda candidate: np.sum(forward_reference(layer, candidate) * upstream))
    np.testing.assert_allclose(gradient, expected, rtol=2e-05, atol=1e-09)

@pytest.mark.parametrize('truncation', [0, 1, 2])
def test_input_gradient_respects_the_existing_truncation_window(truncation):
    layer, x, upstream = make_layer(12, steps=5, truncation=truncation)
    layer.forward_pass(x)
    original_states = layer.states.copy()
    actual = layer.backward_pass(upstream)

    def truncated_objective(candidate):
        loss = 0.0
        for t in range(x.shape[1]):
            start = max(0, t - truncation)
            state = original_states[:, start - 1].copy() if start else np.zeros((x.shape[0], layer.n_units))
            for previous in range(start, t + 1):
                state = np.tanh(candidate[:, previous] @ layer.U.T + state @ layer.W.T)
            loss += np.sum(state @ layer.V.T * upstream[:, t])
        return loss
    expected = differences(x, truncated_objective)
    np.testing.assert_allclose(actual, expected, rtol=2e-05, atol=1e-09)

@pytest.mark.parametrize('steps,zero_recurrence', [(1, False), (4, True)])
def test_nonrecurrent_controls(steps, zero_recurrence):
    layer, x, upstream = make_layer(3, steps)
    if zero_recurrence:
        layer.W[:] = 0
    layer.forward_pass(x)
    actual = layer.backward_pass(upstream)
    expected = differences(x, lambda candidate: np.sum(forward_reference(layer, candidate) * upstream))
    np.testing.assert_allclose(actual, expected, rtol=2e-05, atol=1e-09)

@pytest.mark.parametrize('name', ['U', 'V', 'W'])
def test_parameter_gradients_unchanged_and_correct(name):
    layer, x, upstream = make_layer(4, 3)
    layer.forward_pass(x)
    layer.backward_pass(upstream)
    actual = getattr(layer, name + '_opt').gradient
    original = getattr(layer, name).copy()

    def objective(candidate):
        setattr(layer, name, candidate)
        return np.sum(forward_reference(layer, x) * upstream)
    expected = differences(original, objective)
    setattr(layer, name, original)
    np.testing.assert_allclose(actual, expected, rtol=2e-05, atol=1e-09)

def test_stacked_rnn_chain_rule():
    lower, x, _ = make_layer(6, 4)
    upper, _, upstream = make_layer(8, 4)
    hidden = lower.forward_pass(x)
    upper.forward_pass(hidden)
    actual = lower.backward_pass(upper.backward_pass(upstream))

    def objective(candidate):
        hidden = forward_reference(lower, candidate)
        return np.sum(forward_reference(upper, hidden) * upstream)
    expected = differences(x, objective)
    np.testing.assert_allclose(actual, expected, rtol=2e-05, atol=1e-09)
