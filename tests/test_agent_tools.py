import json

import agent
import model as model_module
import rag
import system_status


def test_execute_tool_predict_churn_returns_result(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "predict_churn",
        lambda cid: {"customer_id": cid, "churn_probability": 0.5, "risk_level": "medium"},
    )
    sources = []
    result = json.loads(agent._execute_tool("predict_churn", {"customer_id": "X1"}, sources))
    assert result["risk_level"] == "medium"


def test_execute_tool_predict_churn_handles_missing_customer(monkeypatch):
    def raise_not_found(cid):
        raise model_module.CustomerNotFound(cid)

    monkeypatch.setattr(model_module, "predict_churn", raise_not_found)
    sources = []
    result = json.loads(agent._execute_tool("predict_churn", {"customer_id": "ghost"}, sources))
    assert "error" in result


def test_execute_tool_explain_prediction_handles_missing_customer(monkeypatch):
    import explain

    def raise_not_found(cid):
        raise model_module.CustomerNotFound(cid)

    monkeypatch.setattr(explain, "explain_churn", raise_not_found)
    sources = []
    result = json.loads(agent._execute_tool("explain_prediction", {"customer_id": "ghost"}, sources))
    assert "error" in result


def test_execute_tool_retrieve_docs_collects_sources(monkeypatch):
    monkeypatch.setattr(
        rag, "retrieve", lambda query, k=3: [{"text": "abc", "source": "doc.md#1"}]
    )
    sources = []
    agent._execute_tool("retrieve_docs", {"query": "test"}, sources)
    assert sources == ["doc.md#1"]


def test_execute_tool_retrieve_docs_does_not_duplicate_sources(monkeypatch):
    monkeypatch.setattr(
        rag, "retrieve", lambda query, k=3: [{"text": "abc", "source": "doc.md#1"}]
    )
    sources = ["doc.md#1"]
    agent._execute_tool("retrieve_docs", {"query": "test"}, sources)
    assert sources == ["doc.md#1"]


def test_execute_tool_unknown_tool_returns_error():
    sources = []
    result = json.loads(agent._execute_tool("made_up_tool", {}, sources))
    assert "error" in result


def test_execute_tool_count_customers_by_risk(monkeypatch):
    monkeypatch.setattr(
        system_status,
        "get_risk_summary",
        lambda: {"total_customers": 100, "risk_counts": {"high": 20, "medium": 30, "low": 50}},
    )
    sources = []
    result = json.loads(agent._execute_tool("count_customers_by_risk", {}, sources))
    assert result["risk_counts"]["high"] == 20
    assert result["total_customers"] == 100


def test_execute_tool_list_high_risk_customers_uses_requested_n(monkeypatch):
    captured = {}

    def fake_top_risk(n):
        captured["n"] = n
        return [{"customer_id": "X1", "churn_probability": 0.9}]

    monkeypatch.setattr(system_status, "get_top_risk_customers", fake_top_risk)
    sources = []
    result = json.loads(agent._execute_tool("list_high_risk_customers", {"n": 5}, sources))
    assert captured["n"] == 5
    assert result[0]["customer_id"] == "X1"


def test_execute_tool_list_high_risk_customers_defaults_to_10(monkeypatch):
    captured = {}

    def fake_top_risk(n):
        captured["n"] = n
        return []

    monkeypatch.setattr(system_status, "get_top_risk_customers", fake_top_risk)
    sources = []
    agent._execute_tool("list_high_risk_customers", {}, sources)
    assert captured["n"] == 10
