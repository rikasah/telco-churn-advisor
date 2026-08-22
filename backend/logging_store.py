"""Logs every /predict and /chat request to Postgres and serves aggregate
stats for the analytics view -- so the demo shows a system that's actually
monitored in production, not a script that gets invoked once and forgotten.
"""
import os

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/churn_db"
)
_engine = create_engine(DATABASE_URL)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS request_logs (
    id SERIAL PRIMARY KEY,
    endpoint TEXT NOT NULL,
    customer_id TEXT,
    churn_probability DOUBLE PRECISION,
    risk_level TEXT,
    tool_calls TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _ensure_table():
    with _engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))


def log_predict(customer_id: str, churn_probability: float, risk_level: str) -> None:
    _ensure_table()
    with _engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO request_logs (endpoint, customer_id, churn_probability, risk_level)
                VALUES ('predict', :cid, :prob, :risk)
                """
            ),
            {"cid": customer_id, "prob": churn_probability, "risk": risk_level},
        )


def log_chat(customer_id: str, tool_calls: list[str]) -> None:
    _ensure_table()
    with _engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO request_logs (endpoint, customer_id, tool_calls)
                VALUES ('chat', :cid, :tools)
                """
            ),
            {"cid": customer_id, "tools": ",".join(tool_calls)},
        )


def get_stats(recent_limit: int = 20) -> dict:
    _ensure_table()
    with _engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM request_logs")).scalar()
        predict_count = conn.execute(
            text("SELECT count(*) FROM request_logs WHERE endpoint = 'predict'")
        ).scalar()
        chat_count = conn.execute(
            text("SELECT count(*) FROM request_logs WHERE endpoint = 'chat'")
        ).scalar()

        risk_rows = conn.execute(
            text(
                """
                SELECT risk_level, count(*) FROM request_logs
                WHERE endpoint = 'predict' AND risk_level IS NOT NULL
                GROUP BY risk_level
                """
            )
        ).all()
        risk_distribution = {row[0]: row[1] for row in risk_rows}

        tool_rows = conn.execute(
            text(
                """
                SELECT tool_calls FROM request_logs
                WHERE endpoint = 'chat' AND tool_calls IS NOT NULL AND tool_calls <> ''
                """
            )
        ).all()
        top_tools: dict[str, int] = {}
        for (tools_csv,) in tool_rows:
            for tool in tools_csv.split(","):
                top_tools[tool] = top_tools.get(tool, 0) + 1

        recent_rows = conn.execute(
            text(
                """
                SELECT endpoint, customer_id, risk_level, tool_calls, created_at
                FROM request_logs ORDER BY created_at DESC LIMIT :n
                """
            ),
            {"n": recent_limit},
        ).all()
        recent = [
            {
                "endpoint": r[0],
                "customer_id": r[1],
                "risk_level": r[2],
                "tool_calls": r[3],
                "created_at": r[4].isoformat() if r[4] else None,
            }
            for r in recent_rows
        ]

    return {
        "total_requests": total,
        "predict_count": predict_count,
        "chat_count": chat_count,
        "risk_level_distribution": risk_distribution,
        "top_tools": top_tools,
        "recent": recent,
    }
