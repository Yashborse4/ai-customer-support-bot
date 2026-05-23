"""Module for managing the relational SQLite database and table selection routing.

Provides functionality to initialize mock schemas, seed records, and dynamically
filter database tables for LangChain SQL agents to handle large schemas.
"""

import logging
import os
import sqlite3
from typing import Dict, List, Set
from langchain_community.utilities.sql_database import SQLDatabase
from src.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "support_records.db")

# Dictionary mapping table names to their semantic descriptions.
TABLE_DESCRIPTIONS: Dict[str, str] = {
    "customers": "Contains customer profiles, contact information, emails, and loyalty tier levels.",
    "products": "Contains catalog of items sold by Acme Corp, including name, category, price, and current stock inventory levels.",
    "orders": "Contains transaction history, purchased product IDs, purchase dates, status, quantity, and total billing amounts.",
    "support_tickets": "Contains support interactions, ticket subjects, status (open/closed), priority, and creation dates.",
    "returns": "Contains records of product returns, reasons for return, refund amounts, and status of returns.",
    "shipping": "Contains shipment tracking details, carrier names, tracking numbers, and shipment statuses."
}

def get_db_uri() -> str:
    """Returns the database URI for SQLAlchemy connection.

    Returns:
        A string URI for connecting to the SQLite database.
    """
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return f"sqlite:///{DB_PATH}"

def initialize_database() -> None:
    """Initializes the SQLite database with schemas and populates mock data if empty.

    This setup models typical relational tables for a customer support bot.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. Customers Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                loyalty_tier TEXT NOT NULL CHECK(loyalty_tier IN ('Bronze', 'Silver', 'Gold', 'Platinum'))
            )
        """)

        # 2. Products Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                stock_quantity INTEGER NOT NULL,
                description TEXT
            )
        """)

        # 3. Orders Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                order_date TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Pending', 'Shipped', 'Delivered', 'Cancelled')),
                quantity INTEGER NOT NULL,
                total_amount REAL NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)

        # 4. Support Tickets Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Open', 'In Progress', 'Resolved')),
                priority TEXT NOT NULL CHECK(priority IN ('Low', 'Medium', 'High', 'Critical')),
                created_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )
        """)

        # 5. Returns Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS returns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER UNIQUE NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Processing', 'Approved', 'Rejected')),
                refund_amount REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id)
            )
        """)

        # 6. Shipping Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shipping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER UNIQUE NOT NULL,
                carrier TEXT NOT NULL,
                tracking_number TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Manifested', 'In Transit', 'Out for Delivery', 'Delivered')),
                FOREIGN KEY (order_id) REFERENCES orders(id)
            )
        """)

        conn.commit()

        # Seed mock data if tables are empty
        cursor.execute("SELECT COUNT(*) FROM customers")
        if cursor.fetchone()[0] == 0:
            logger.info("Seeding relational database tables with mock data...")

            # Customers
            customers_data = [
                ("Alice Johnson", "alice.j@example.com", "Gold"),
                ("Bob Smith", "bob.smith@example.com", "Silver"),
                ("Charlie Brown", "charlie.b@example.com", "Bronze"),
                ("Diana Prince", "diana.p@example.com", "Platinum")
            ]
            cursor.executemany("INSERT INTO customers (name, email, loyalty_tier) VALUES (?, ?, ?)", customers_data)

            # Products
            products_data = [
                ("SuperWidget 3000", 129.99, 45, "Flagship smart home controller with multi-protocol support."),
                ("SmartPlug Lite", 24.99, 150, "Energy-monitoring Wi-Fi smart plug."),
                ("Acme SoundBar", 199.99, 15, "Dolby Atmos enabled home theater soundbar."),
                ("Vision Camera", 89.99, 0, "Outdoor 2K security camera with night vision.")
            ]
            cursor.executemany("INSERT INTO products (name, price, stock_quantity, description) VALUES (?, ?, ?, ?)", products_data)

            # Orders
            orders_data = [
                (1, 1, "2026-05-10", "Delivered", 1, 129.99), # Alice bought SuperWidget
                (1, 2, "2026-05-12", "Delivered", 2, 49.98),  # Alice bought SmartPlugs
                (2, 3, "2026-05-20", "Shipped", 1, 199.99),   # Bob bought SoundBar
                (3, 4, "2026-05-21", "Pending", 1, 89.99),    # Charlie bought Vision Camera
                (4, 1, "2026-05-22", "Cancelled", 1, 129.99)  # Diana order cancelled
            ]
            cursor.executemany("INSERT INTO orders (customer_id, product_id, order_date, status, quantity, total_amount) VALUES (?, ?, ?, ?, ?, ?)", orders_data)

            # Support Tickets
            tickets_data = [
                (1, "How to reset SuperWidget 3000", "Resolved", "Medium", "2026-05-11"),
                (2, "Soundbar connection issue", "In Progress", "High", "2026-05-21"),
                (3, "Vision Camera out of stock", "Open", "Low", "2026-05-22")
            ]
            cursor.executemany("INSERT INTO support_tickets (customer_id, subject, status, priority, created_at) VALUES (?, ?, ?, ?, ?)", tickets_data)

            # Returns
            returns_data = [
                (5, "Cancelled before shipment", "Approved", 129.99)
            ]
            cursor.executemany("INSERT INTO returns (order_id, reason, status, refund_amount) VALUES (?, ?, ?, ?)", returns_data)

            # Shipping
            shipping_data = [
                (1, "FedEx", "1Z999AA10123456784", "Delivered"),
                (2, "UPS", "1Z999AA10123456789", "Delivered"),
                (3, "DHL", "DHL8872635412", "In Transit")
            ]
            cursor.executemany("INSERT INTO shipping (order_id, carrier, tracking_number, status) VALUES (?, ?, ?, ?)", shipping_data)

            conn.commit()
            logger.info("Database seeding complete.")

    except sqlite3.Error as e:
        logger.error("SQLite initialization error: %s", e)
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_db_for_query(query: str) -> SQLDatabase:
    """Selects relevant tables for a given query and returns a SQLDatabase instance.

    This implements table grouping/routing to avoid overwhelming the LLM context.
    If a query targets specific topics (e.g. returns, orders), only the tables
    associated with those concepts are loaded. If no specific tables match,
    a core default group (customers, orders, products) is used.

    Args:
        query: User's question or search query.

    Returns:
        An initialized LangChain SQLDatabase instance with filtered tables.
    """
    # Ensure database is initialized
    initialize_database()

    query_lower = query.lower()
    selected_tables: Set[str] = set()

    # Rule-based table routing maps keyword indicators to table names
    keyword_mapping: Dict[str, List[str]] = {
        "customer": ["customers", "orders", "support_tickets"],
        "user": ["customers"],
        "client": ["customers"],
        "loyalty": ["customers"],
        "order": ["orders", "shipping", "returns"],
        "purchase": ["orders"],
        "buy": ["orders", "products"],
        "bought": ["orders", "products"],
        "product": ["products", "orders"],
        "item": ["products"],
        "stock": ["products"],
        "inventory": ["products"],
        "price": ["products"],
        "cost": ["products"],
        "ticket": ["support_tickets", "customers"],
        "issue": ["support_tickets"],
        "support": ["support_tickets"],
        "complain": ["support_tickets"],
        "return": ["returns", "orders"],
        "refund": ["returns", "orders"],
        "ship": ["shipping", "orders"],
        "delivery": ["shipping", "orders"],
        "track": ["shipping"],
        "carrier": ["shipping"],
        "fedex": ["shipping"],
        "ups": ["shipping"],
        "dhl": ["shipping"]
    }

    for keyword, tables in keyword_mapping.items():
        if keyword in query_lower:
            selected_tables.update(tables)

    # Default fallback tables if no keywords matched
    if not selected_tables:
        selected_tables = {"customers", "orders", "products"}

    # Include dependencies for foreign key lookups if necessary
    if "returns" in selected_tables or "shipping" in selected_tables:
        selected_tables.add("orders")
    if "orders" in selected_tables:
        selected_tables.add("customers")
        selected_tables.add("products")

    selected_list = list(selected_tables)
    logger.info("Routing query '%s' to database tables: %s", query, selected_list)

    # Initialize SQLDatabase with specific tables
    return SQLDatabase.from_uri(
        get_db_uri(),
        include_tables=selected_list
    )
