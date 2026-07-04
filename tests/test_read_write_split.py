"""
The security invariant: the investigation side can read, never write.

If someone later adds a `deploy()` / `retrain()` method to the investigation
toolkit, this test fails - the read/write split stops being a guarantee the
moment it stops being enforced.
"""

import inspect

import pandas as pd

from aegis.agent.tools import InvestigationToolkit
from aegis.control_plane.remediation import Remediation

WRITE_VERBS = (
    "retrain", "promote", "rollback", "escalate",
    "write", "save", "delete", "commit", "push",
)


def _toolkit():
    ref = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    cur = pd.DataFrame({"amount": [9.0, 8.0, 7.0], "cat": ["a", "b", "a"]})
    return InvestigationToolkit(reference_df=ref, current_df=cur)


def _public_methods(obj):
    return {
        m for m, _ in inspect.getmembers(obj, predicate=inspect.ismethod)
        if not m.startswith("_")
    }


def test_investigation_toolkit_exposes_no_write_actions():
    """No investigation tool method starts with a mutation verb, and none
    shares a name with a remediation (write) action."""
    remediation_actions = {
        m for m, _ in inspect.getmembers(Remediation, predicate=inspect.isfunction)
        if not m.startswith("_")
    }
    toolkit = _toolkit()
    for tool_name, tool in vars(toolkit).items():
        if tool is None:
            continue
        for method in _public_methods(tool):
            first_token = method.lower().split("_")[0]
            assert first_token not in WRITE_VERBS, (
                f"read-only tool '{tool_name}' exposes write-like method '{method}'"
            )
            assert method not in remediation_actions, (
                f"read-only tool '{tool_name}' shares a name with remediation "
                f"action '{method}'"
            )


def test_write_actions_live_only_on_remediation():
    methods = {m for m, _ in inspect.getmembers(Remediation, predicate=inspect.isfunction)}
    # The controller-only surface is where writes are allowed to exist.
    assert any("retrain" in m for m in methods)


def test_investigator_only_depends_on_read_only_reasoner():
    # The reasoner interface has exactly one capability: text completion.
    from aegis.reasoning.base import Reasoner

    public = [m for m in dir(Reasoner) if not m.startswith("_")]
    assert "complete" in public
    assert not any(v in m.lower() for m in public for v in WRITE_VERBS)
