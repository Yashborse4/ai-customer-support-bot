"""Module for managing the relational SQLite database and table selection routing.

Provides functionality to initialize mock schemas, seed records, and dynamically
filter database tables for LangChain SQL agents to handle large schemas.
"""

import logging
import os
from typing import Dict, List, Set, Optional, Any
from langchain_community.utilities.sql_database import SQLDatabase
from src.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Default semantic descriptions for SQL tables to guide query agent routing
TABLE_DESCRIPTIONS: Dict[str, str] = {
    "customers": "Contains customer details including names, email addresses, phone numbers, and loyalty tiers.",
    "products": "Contains details of company products, descriptions, prices, stock quantities, and categories.",
    "orders": "Contains transaction records linking customers and products, purchase dates, order totals, and statuses.",
    "support_tickets": "Contains user customer support history, problem descriptions, priority rankings, status flags, and timestamps.",
    "returns": "Contains customer refund records, return reasons, eligibility statuses, and processing dates.",
    "shipping": "Contains package delivery details, carrier names, tracking numbers, estimated arrival dates, and shipping progress status."
}

def get_db_credentials() -> Dict[str, Any]:
    """Loads database connection credentials, falling back to settings."""
    config_path = os.path.join("data", "db_config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return {
                    "user": config.get("user", settings.DB_USER),
                    "password": config.get("password", settings.DB_PASSWORD),
                    "host": config.get("host", settings.DB_HOST),
                    "port": int(config.get("port", settings.DB_PORT)),
                    "service_name": config.get("service_name", settings.DB_SERVICE_NAME)
                }
        except Exception as e:
            logger.error("Failed to load db_config.json: %s", e)
    return {
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
        "host": settings.DB_HOST,
        "port": settings.DB_PORT,
        "service_name": settings.DB_SERVICE_NAME
    }

def get_db_uri() -> str:
    """Returns the database URI for SQLAlchemy connection.

    Returns:
        A string URI for connecting to the Oracle database.
    """
    creds = get_db_credentials()
    return f"oracle+oracledb://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['service_name']}"

def get_oracle_tables() -> List[str]:
    """Retrieves all user table names from Oracle.

    Returns:
        A list of table name strings.
    """
    import oracledb
    creds = get_db_credentials()
    try:
        conn = oracledb.connect(
            user=creds["user"],
            password=creds["password"],
            host=creds["host"],
            port=creds["port"],
            service_name=creds["service_name"]
        )
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
        tables = [row[0].lower() for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return tables
    except Exception as e:
        logger.error("Failed to fetch Oracle tables: %s", e)
        return []

def initialize_database() -> None:
    """Initializes the Oracle database with schemas and populates mock data if empty.

    This setup models typical relational tables for a customer support bot.
    """
    import oracledb
    creds = get_db_credentials()

    try:
        # Try to connect to Oracle 11g database using thin mode
        conn = oracledb.connect(
            user=creds["user"],
            password=creds["password"],
            host=creds["host"],
            port=creds["port"],
            service_name=creds["service_name"]
        )
    except Exception as e:
        logger.warning(
            "Could not connect to Oracle database: %s. Skipping automatic table initialization. "
            "Please ensure Oracle 11g is running and credentials are correct in your .env.", e
        )
        return

    cursor = conn.cursor()

    def table_exists(table_name: str) -> bool:
        cursor.execute("SELECT COUNT(*) FROM user_tables WHERE table_name = :1", (table_name.upper(),))
        return cursor.fetchone()[0] > 0

    try:
        # 1. Customers Table
        if not table_exists("customers"):
            cursor.execute("""
                CREATE TABLE customers (
                    id NUMBER PRIMARY KEY,
                    name VARCHAR2(150) NOT NULL,
                    email VARCHAR2(150) UNIQUE NOT NULL,
                    loyalty_tier VARCHAR2(50) NOT NULL CHECK(loyalty_tier IN ('Bronze', 'Silver', 'Gold', 'Platinum'))
                )
            """)

        # 2. Products Table
        if not table_exists("products"):
            cursor.execute("""
                CREATE TABLE products (
                    id NUMBER PRIMARY KEY,
                    name VARCHAR2(150) NOT NULL,
                    price NUMBER NOT NULL,
                    stock_quantity NUMBER NOT NULL,
                    description VARCHAR2(1000)
                )
            """)

        # 3. Orders Table
        if not table_exists("orders"):
            cursor.execute("""
                CREATE TABLE orders (
                    id NUMBER PRIMARY KEY,
                    customer_id NUMBER NOT NULL,
                    product_id NUMBER NOT NULL,
                    order_date VARCHAR2(50) NOT NULL,
                    status VARCHAR2(50) NOT NULL CHECK(status IN ('Pending', 'Shipped', 'Delivered', 'Cancelled')),
                    quantity NUMBER NOT NULL,
                    total_amount NUMBER NOT NULL,
                    FOREIGN KEY (customer_id) REFERENCES customers(id),
                    FOREIGN KEY (product_id) REFERENCES products(id)
                )
            """)

        # 4. Support Tickets Table
        if not table_exists("support_tickets"):
            cursor.execute("""
                CREATE TABLE support_tickets (
                    id NUMBER PRIMARY KEY,
                    customer_id NUMBER NOT NULL,
                    subject VARCHAR2(500) NOT NULL,
                    status VARCHAR2(50) NOT NULL CHECK(status IN ('Open', 'In Progress', 'Resolved')),
                    priority VARCHAR2(50) NOT NULL CHECK(priority IN ('Low', 'Medium', 'High', 'Critical')),
                    created_at VARCHAR2(50) NOT NULL,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
            """)

        # 5. Returns Table
        if not table_exists("returns"):
            cursor.execute("""
                CREATE TABLE returns (
                    id NUMBER PRIMARY KEY,
                    order_id NUMBER UNIQUE NOT NULL,
                    reason VARCHAR2(500) NOT NULL,
                    status VARCHAR2(50) NOT NULL CHECK(status IN ('Processing', 'Approved', 'Rejected')),
                    refund_amount NUMBER NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(id)
                )
            """)

        # 6. Shipping Table
        if not table_exists("shipping"):
            cursor.execute("""
                CREATE TABLE shipping (
                    id NUMBER PRIMARY KEY,
                    order_id NUMBER UNIQUE NOT NULL,
                    carrier VARCHAR2(50) NOT NULL,
                    tracking_number VARCHAR2(100) UNIQUE NOT NULL,
                    status VARCHAR2(50) NOT NULL CHECK(status IN ('Manifested', 'In Transit', 'Out for Delivery', 'Delivered')),
                    FOREIGN KEY (order_id) REFERENCES orders(id)
                )
            """)

        conn.commit()

        # Seed mock data if tables are empty
        cursor.execute("SELECT COUNT(*) FROM customers")
        if cursor.fetchone()[0] == 0:
            logger.info("Seeding Oracle database tables with mock data...")

            # Customers
            customers_data = [
                (1, "Alice Johnson", "alice.j@example.com", "Gold"),
                (2, "Bob Smith", "bob.smith@example.com", "Silver"),
                (3, "Charlie Brown", "charlie.b@example.com", "Bronze"),
                (4, "Diana Prince", "diana.p@example.com", "Platinum")
            ]
            for row in customers_data:
                cursor.execute("INSERT INTO customers (id, name, email, loyalty_tier) VALUES (:1, :2, :3, :4)", row)

            # Products
            products_data = [
                (1, "SuperWidget 3000", 129.99, 45, "Flagship smart home controller with multi-protocol support."),
                (2, "SmartPlug Lite", 24.99, 150, "Energy-monitoring Wi-Fi smart plug."),
                (3, "Acme SoundBar", 199.99, 15, "Dolby Atmos enabled home theater soundbar."),
                (4, "Vision Camera", 89.99, 0, "Outdoor 2K security camera with night vision.")
            ]
            for row in products_data:
                cursor.execute("INSERT INTO products (id, name, price, stock_quantity, description) VALUES (:1, :2, :3, :4, :5)", row)

            # Orders
            orders_data = [
                (1, 1, 1, "2026-05-10", "Delivered", 1, 129.99), # Alice bought SuperWidget
                (2, 1, 2, "2026-05-12", "Delivered", 2, 49.98),  # Alice bought SmartPlugs
                (3, 2, 3, "2026-05-20", "Shipped", 1, 199.99),   # Bob bought SoundBar
                (4, 3, 4, "2026-05-21", "Pending", 1, 89.99),    # Charlie bought Vision Camera
                (5, 4, 1, "2026-05-22", "Cancelled", 1, 129.99)  # Diana order cancelled
            ]
            for row in orders_data:
                cursor.execute("INSERT INTO orders (id, customer_id, product_id, order_date, status, quantity, total_amount) VALUES (:1, :2, :3, :4, :5, :6, :7)", row)

            # Support Tickets
            tickets_data = [
                (1, 1, "How to reset SuperWidget 3000", "Resolved", "Medium", "2026-05-11"),
                (2, 2, "Soundbar connection issue", "In Progress", "High", "2026-05-21"),
                (3, 3, "Vision Camera out of stock", "Open", "Low", "2026-05-22")
            ]
            for row in tickets_data:
                cursor.execute("INSERT INTO support_tickets (id, customer_id, subject, status, priority, created_at) VALUES (:1, :2, :3, :4, :5, :6)", row)

            # Returns
            returns_data = [
                (1, 5, "Cancelled before shipment", "Approved", 129.99)
            ]
            for row in returns_data:
                cursor.execute("INSERT INTO returns (id, order_id, reason, status, refund_amount) VALUES (:1, :2, :3, :4, :5)", row)

            # Shipping
            shipping_data = [
                (1, 1, "FedEx", "1Z999AA10123456784", "Delivered"),
                (2, 2, "UPS", "1Z999AA10123456789", "Delivered"),
                (3, 3, "DHL", "DHL8872635412", "In Transit")
            ]
            for row in shipping_data:
                cursor.execute("INSERT INTO shipping (id, order_id, carrier, tracking_number, status) VALUES (:1, :2, :3, :4, :5)", row)

            conn.commit()
            logger.info("Oracle database seeding complete.")

    except oracledb.Error as e:
        logger.error("Oracle initialization error: %s", e)
        conn.rollback()
        raise e
    finally:
        conn.close()

# Allowed tables scoped per department to satisfy enterprise segregation requirements.
DEPARTMENT_TABLE_SCOPES: Dict[str, Set[str]] = {
    "sales": {"customers", "orders", "products", "shipping"},
    "technical": {"products", "support_tickets"},
    "billing": {"customers", "orders", "returns"},
    "general": {"customers", "orders", "products", "support_tickets", "returns", "shipping"}
}

def get_db_for_query(query: str, department: Optional[str] = None) -> SQLDatabase:
    """Selects relevant tables for a given query and returns a SQLDatabase instance.

    This implements table grouping/routing to avoid overwhelming the LLM context.
    It filters the allowed tables based on the user's department for access control.

    Args:
        query: User's question or search query.
        department: Optional department name to restrict database scopes.

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

    # Load dynamic table metadata if exists and perform semantic description matching
    table_metadata_path = os.path.join("data", "table_metadata.json")
    table_desc = TABLE_DESCRIPTIONS.copy()
    if os.path.exists(table_metadata_path):
        try:
            import json
            with open(table_metadata_path, "r", encoding="utf-8") as f:
                saved_metadata = json.load(f)
                for tbl, desc in saved_metadata.items():
                    table_desc[tbl.lower()] = desc
        except Exception as e:
            logger.error("Failed to load table_metadata.json: %s", e)

    # Match user query terms against semantic descriptions
    for tbl, desc in table_desc.items():
        desc_lower = desc.lower()
        for word in query_lower.split():
            if len(word) > 3 and word in desc_lower:
                selected_tables.add(tbl)

    # Default fallback tables if no keywords matched
    if not selected_tables:
        selected_tables = {"customers", "orders", "products"}

    # Include dependencies for foreign key lookups if necessary
    if "returns" in selected_tables or "shipping" in selected_tables:
        selected_tables.add("orders")
    if "orders" in selected_tables:
        selected_tables.add("customers")
        selected_tables.add("products")

    # Apply department-level table filtering for segregation
    if department:
        dept_key = department.lower()
        allowed_tables = DEPARTMENT_TABLE_SCOPES.get(dept_key, DEPARTMENT_TABLE_SCOPES["general"])
        selected_tables = selected_tables.intersection(allowed_tables)
        if not selected_tables:
            # Fallback to a safe allowed subset
            selected_tables = allowed_tables.intersection({"customers", "orders", "products", "support_tickets"})

    selected_list = list(selected_tables)
    logger.info("Routing query '%s' (dept: %s) to database tables: %s", query, department, selected_list)

    # Initialize SQLDatabase with specific tables
    return SQLDatabase.from_uri(
        get_db_uri(),
        include_tables=selected_list
    )

def test_db_connection() -> bool:
    """Tests connection to the Oracle database.

    Returns:
        True if connection succeeds, False otherwise.
    """
    import oracledb
    creds = get_db_credentials()
    try:
        conn = oracledb.connect(
            user=creds["user"],
            password=creds["password"],
            host=creds["host"],
            port=creds["port"],
            service_name=creds["service_name"]
        )
        conn.close()
        return True
    except Exception as e:
        logger.error("Database connection test failed: %s", e)
        return False
