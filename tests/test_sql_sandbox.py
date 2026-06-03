"""Unit tests for the SQL Sandbox guardrails in database query tools."""

import pytest
from sqlalchemy import create_engine, text
from src.database.sql_db import register_sandbox_guardrails

@pytest.fixture
def test_engine():
    """Creates an in-memory SQLite database engine with sandbox guardrails registered."""
    engine = create_engine("sqlite:///:memory:")
    
    # Initialize some mock tables to test with write queries
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO test_table (id, name) VALUES (1, 'Alice')"))
        conn.commit()
        
    # Register sandbox guardrails to intercept all database queries
    register_sandbox_guardrails(engine)
    yield engine

def test_select_query_allowed(test_engine) -> None:
    """Verify that SELECT statements execute successfully in the sandbox."""
    with test_engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM test_table WHERE id = 1")).fetchone()
        assert result is not None
        assert result[0] == "Alice"

@pytest.mark.parametrize("query", [
    "INSERT INTO test_table (id, name) VALUES (2, 'Bob')",
    "UPDATE test_table SET name = 'Charlie' WHERE id = 1",
    "DELETE FROM test_table WHERE id = 1",
    "DROP TABLE test_table",
    "ALTER TABLE test_table ADD COLUMN description TEXT",
    "CREATE TABLE new_table (id INTEGER)",
    "REPLACE INTO test_table (id, name) VALUES (1, 'Dave')",
    "TRUNCATE TABLE test_table"
])
def test_write_queries_blocked(test_engine, query: str) -> None:
    """Verify that forbidden write statements raise a PermissionError and are blocked."""
    with test_engine.connect() as conn:
        with pytest.raises(PermissionError) as exc_info:
            conn.execute(text(query))
        
        assert "Security Alert" in str(exc_info.value)
        assert "blocked" in str(exc_info.value)
