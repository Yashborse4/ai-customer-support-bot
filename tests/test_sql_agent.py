"""Unit and integration tests for SQLite database, query routing, and SQL Agent."""

import os
import pytest
from src.database.sql_db import initialize_database, get_db_for_query, DB_PATH
from src.tools.sql_tool import query_customer_database

def test_database_initialization() -> None:
    """Verify that initialize_database creates the file and populates the schema."""
    # Ensure any previous test database is removed to test fresh setup
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except PermissionError:
            pass

    initialize_database()
    assert os.path.exists(DB_PATH), "Database file should be created."

    # Validate querying the database works
    db = get_db_for_query("Show me all products")
    tables = db.get_usable_table_names()
    
    # Selected tables for product keyword
    assert "products" in tables
    assert "orders" in tables

def test_table_selection_routing() -> None:
    """Verify that get_db_for_query filters tables based on search keywords."""
    # Returns/refund queries
    db_returns = get_db_for_query("I want to return an item and get a refund")
    tables_returns = db_returns.get_usable_table_names()
    assert "returns" in tables_returns
    assert "orders" in tables_returns
    assert "customers" in tables_returns
    assert "products" in tables_returns
    assert "shipping" not in tables_returns

    # Shipping/delivery queries
    db_shipping = get_db_for_query("Where is my tracking number for UPS?")
    tables_shipping = db_shipping.get_usable_table_names()
    assert "shipping" in tables_shipping
    assert "orders" in tables_shipping
    assert "returns" not in tables_shipping

    # Fallback default queries
    db_default = get_db_for_query("Hello there")
    tables_default = db_default.get_usable_table_names()
    assert "customers" in tables_default
    assert "orders" in tables_default
    assert "products" in tables_default
    assert "shipping" not in tables_default

def test_department_segregation() -> None:
    """Verify that get_db_for_query restricts database scopes by department."""
    # Technical department should only see products and support tickets
    db_tech = get_db_for_query("Show me Alice's orders and tickets", department="technical")
    tables_tech = db_tech.get_usable_table_names()
    assert "support_tickets" in tables_tech
    assert "products" in tables_tech
    assert "orders" not in tables_tech
    assert "customers" not in tables_tech

    # Billing department should only see customers, orders, and returns
    db_billing = get_db_for_query("Show me Bob's orders and shipping status", department="billing")
    tables_billing = db_billing.get_usable_table_names()
    assert "orders" in tables_billing
    assert "customers" in tables_billing
    assert "returns" in tables_billing
    assert "shipping" not in tables_billing

from unittest.mock import patch, MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_query_customer_database_tool() -> None:
    """Verify that the query_customer_database tool executes successfully using a mock."""
    mock_responses = {
        "What is the loyalty tier of customer Alice Johnson?": {"output": "Customer Alice Johnson has a Gold loyalty tier."},
        "What is the stock quantity of SuperWidget 3000?": {"output": "The stock quantity of SuperWidget 3000 is 45."}
    }

    async def mock_ainvoke(inputs, *args, **kwargs):
        query = inputs.get("input", "")
        return mock_responses.get(query, {"output": "Mocked response."})

    with patch("src.tools.sql_tool.create_sql_agent") as mock_create:
        mock_executor = MagicMock()
        mock_executor.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        mock_create.return_value = mock_executor

        # Query for Alice's loyalty tier
        response = await query_customer_database.ainvoke({"query": "What is the loyalty tier of customer Alice Johnson?"})
        assert "Gold" in response, f"Expected Gold in response, got: {response}"

        # Query for SuperWidget stock quantity
        response_stock = await query_customer_database.ainvoke({"query": "What is the stock quantity of SuperWidget 3000?"})
        assert "45" in response_stock, f"Expected 45 in response, got: {response_stock}"

