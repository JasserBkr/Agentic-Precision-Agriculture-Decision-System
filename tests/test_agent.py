import pytest

from agri_agent.agent.graph import agent_node


def test_agent_node_not_yet_implemented():
    # TODO (Week 5-6): replace once agent/graph.py is built.
    with pytest.raises(NotImplementedError):
        agent_node({})
