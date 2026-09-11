import torch

from sar.methods.fixed_kv_subspace import FixedKVSubspace
from sar.methods.head_router import HeadRouter
from sar.methods.layer_router import LayerRouter
from sar.methods.oracle_mask import OracleMaskRouter
from sar.methods.static_gate import StaticAudioGate
from sar.models.base import RoutingContext


def context():
    return RoutingContext(
        query_repr=torch.tensor([[1.0, 0.0]]),
        audio_repr=torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]]]),
        audio_mask=torch.tensor([[True, True, True]]),
        relevance_mask=torch.tensor([[True, False, True]]),
    )


def test_static_gate_is_query_invariant():
    method = StaticAudioGate(initial_gate=0.7)
    a = method.route(context()).gate
    c = context(); c.query_repr = torch.tensor([[-4.0, 2.0]])
    b = method.route(c).gate
    assert torch.allclose(a, b)


def test_layer_router_outputs_valid_weights():
    method = LayerRouter(query_dim=2, num_layers=3)
    w = method.layer_weights(context().query_repr)
    assert w.shape == (1, 3)
    assert torch.all((0 <= w) & (w <= 1))


def test_head_router_outputs_valid_weights():
    method = HeadRouter(query_dim=2, num_heads=4)
    w = method.head_weights(context().query_repr)
    assert w.shape == (1, 4)
    assert torch.all((0 <= w) & (w <= 1))


def test_oracle_router_uses_relevance_mask():
    g = OracleMaskRouter().route(context()).gate
    assert torch.equal(g.bool(), context().relevance_mask)


def test_fixed_kv_subspace_is_separate_projection_control():
    method = FixedKVSubspace(torch.tensor([[1.0], [0.0]]), strength=1.0)
    x = torch.tensor([[2.0, 3.0]])
    y = method.project(x)
    assert torch.allclose(y, torch.tensor([[0.0, 3.0]]))
