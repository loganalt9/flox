from __future__ import annotations

import typing
from dataclasses import dataclass, field

from proxystore.proxy import Proxy

if typing.TYPE_CHECKING:
    from pandas import DataFrame

    from flox.flock import NodeID, NodeKind
    from flox.flock.states import NodeState
    from flox.nn.typing import StateDict


@dataclass
class JobResult:
    """A simple dataclass that is returned by jobs executed on Aggregator and Worker nodes in a ``Flock``.

    Aggregators and Worker nodes have to return the same type of object to support hierarchical execution.
    """

    node_state: NodeState
    """The state of the ``Flock`` node based on its kind."""

    node_idx: NodeID
    """The ID of the ``Flock`` node."""

    node_kind: NodeKind
    """The kind of the ``Flock`` node."""

    state_dict: StateDict
    """The ``StateDict`` of the PyTorch global_module (either aggregated or trained locally)."""

    history: DataFrame
    """The history of results."""

    cache: dict[str, typing.Any] = field(default_factory=dict)
    """Miscellaneous data to be returned as part of the ``JobResult``."""


Result: typing.TypeAlias = JobResult | Proxy[JobResult]
