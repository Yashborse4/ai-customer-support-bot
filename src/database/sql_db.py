"""Module for managing the relational SQLite/Oracle database and table selection routing.

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

METADATA_DB_PATH = "data/metadata.db"

def initialize_metadata_db() -> None:
    """Initializes the SQLite database used to store table metadata and system settings."""
    import sqlite3
    os.makedirs(os.path.dirname(METADATA_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(METADATA_DB_PATH)
    cursor = conn.cursor()
    try:
        # Table metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS table_metadata (
                table_name TEXT PRIMARY KEY,
                description TEXT NOT NULL
            )
        """)
        
        # System settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()
        
        # Populate default descriptions if empty
        cursor.execute("SELECT COUNT(*) FROM table_metadata")
        if cursor.fetchone()[0] == 0:
            for tbl, desc in TABLE_DESCRIPTIONS.items():
                cursor.execute(
                    "INSERT INTO table_metadata (table_name, description) VALUES (?, ?)",
                    (tbl.lower(), desc)
                )
            conn.commit()
            
        # Seed default settings from environment/Pydantic configurations if empty
        cursor.execute("SELECT COUNT(*) FROM system_settings")
        if cursor.fetchone()[0] == 0:
            default_settings = {
                "db_type": settings.DB_TYPE,
                "sqlite_db_path": settings.SQLITE_DB_PATH,
                "db_user": settings.DB_USER,
                "db_password": settings.DB_PASSWORD,
                "db_host": settings.DB_HOST,
                "db_port": str(settings.DB_PORT),
                "db_service_name": settings.DB_SERVICE_NAME,
                "model_name": settings.MODEL_NAME,
                "embedding_model": settings.EMBEDDING_MODEL,
                "local_llm_base_url": settings.LOCAL_LLM_BASE_URL,
                "local_embedding_base_url": settings.LOCAL_EMBEDDING_BASE_URL,
                "vector_db_type": settings.VECTOR_DB_TYPE
            }
            for k, v in default_settings.items():
                cursor.execute(
                    "INSERT INTO system_settings (key, value) VALUES (?, ?)",
                    (k, str(v))
                )
            conn.commit()
    except Exception as e:
        logger.error("Failed to initialize metadata DB: %s", e)
    finally:
        conn.close()

def get_all_table_metadata() -> Dict[str, str]:
    """Retrieves all table descriptions from the SQLite metadata database."""
    initialize_metadata_db()
    import sqlite3
    metadata = {}
    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT table_name, description FROM table_metadata")
        for row in cursor.fetchall():
            metadata[row[0].lower()] = row[1]
        conn.close()
    except Exception as e:
        logger.error("Failed to read table metadata from SQLite: %s", e)
        return TABLE_DESCRIPTIONS.copy()
    return metadata

def save_table_metadata_db(table_name: str, description: str) -> None:
    """Saves/updates a table's semantic description inside the SQLite metadata database."""
    initialize_metadata_db()
    import sqlite3
    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO table_metadata (table_name, description) VALUES (?, ?)",
            (table_name.lower(), description)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to save table metadata to SQLite: %s", e)
        raise e

def get_system_setting(key: str, default: Any = None) -> Any:
    """Retrieves a system setting value from the SQLite metadata database."""
    initialize_metadata_db()
    import sqlite3
    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key = ?", (key.lower(),))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception as e:
        logger.error("Failed to read system setting %s: %s", key, e)
    return default

def save_system_setting(key: str, value: str) -> None:
    """Saves or updates a system setting in the SQLite metadata database."""
    initialize_metadata_db()
    import sqlite3
    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)",
            (key.lower(), str(value))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to save system setting %s: %s", key, e)
        raise e

def load_settings_from_db() -> None:
    """Loads all system settings from SQLite database and overrides settings singleton."""
    initialize_metadata_db()
    import sqlite3
    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM system_settings")
        rows = cursor.fetchall()
        conn.close()
        
        for key, val in rows:
            # Map database keys to settings attributes
            attr = key.upper()
            if hasattr(settings, attr):
                # Cast port to int
                if attr == "DB_PORT":
                    setattr(settings, attr, int(val))
                else:
                    setattr(settings, attr, val)
                logger.info("Configuration override from database: %s = %s", attr, val)
    except Exception as e:
        logger.error("Failed to load settings from DB: %s", e)

def get_db_credentials() -> Dict[str, Any]:
    """Loads database connection credentials from settings."""
    return {
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
        "host": settings.DB_HOST,
        "port": settings.DB_PORT,
        "service_name": settings.DB_SERVICE_NAME
    }

def get_oracle_dsn(creds: Dict[str, Any]) -> str:
    """Constructs the Oracle Data Source Name (DSN) from connection credentials.

    Supports:
    1. Full connection descriptors / TNS strings (e.g. starting with '(').
    2. Standard host:port/service_name strings.
    """
    host = creds.get("host", "").strip()
    if host.startswith("("):
        return host
    port = creds.get("port", 1521)
    service_name = creds.get("service_name", "xe")
    return f"{host}:{port}/{service_name}"

def get_db_uri() -> str:
    """Returns the database URI for SQLAlchemy connection.

    Returns:
        A string URI for connecting to the database.
    """
    if settings.DB_TYPE.lower() == "sqlite":
        os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH) or "data", exist_ok=True)
        return f"sqlite:///{settings.SQLITE_DB_PATH}"
        
    creds = get_db_credentials()
    dsn = get_oracle_dsn(creds)
    if dsn.startswith("("):
        import urllib.parse
        encoded_dsn = urllib.parse.quote_plus(dsn)
        return f"oracle+oracledb://{creds['user']}:{creds['password']}@/?dsn={encoded_dsn}"
    return f"oracle+oracledb://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['service_name']}"

def get_db_tables() -> List[str]:
    """Retrieves all user table names from the database (SQLite or Oracle).

    Returns:
        A list of table name strings.
    """
    if settings.DB_TYPE.lower() == "sqlite":
        import sqlite3
        db_path = settings.SQLITE_DB_PATH
        if not os.path.exists(db_path):
            return []
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
            tables = [row[0].lower() for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return tables
        except Exception as e:
            logger.error("Failed to fetch SQLite tables: %s", e)
            return []

    # Oracle mode
    import oracledb
    creds = get_db_credentials()
    try:
        conn = oracledb.connect(
            user=creds["user"],
            password=creds["password"],
            dsn=get_oracle_dsn(creds)
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

# Backwards compatibility alias
get_oracle_tables = get_db_tables

def initialize_sqlite() -> None:
    """Initializes the local SQLite database with tables and mock seed data if empty."""
    import sqlite3
    db_path = settings.SQLITE_DB_PATH
    
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    try:
        # Customers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                loyalty_tier TEXT NOT NULL CHECK(loyalty_tier IN ('Bronze', 'Silver', 'Gold', 'Platinum'))
            )
        """)
        
        # Products
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                stock_quantity INTEGER NOT NULL,
                description TEXT
            )
        """)
        
        # Orders
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
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
        
        # Support Tickets
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Open', 'In Progress', 'Resolved')),
                priority TEXT NOT NULL CHECK(priority IN ('Low', 'Medium', 'High', 'Critical')),
                created_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )
        """)
        
        # Returns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS returns (
                id INTEGER PRIMARY KEY,
                order_id INTEGER UNIQUE NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Processing', 'Approved', 'Rejected')),
                refund_amount REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id)
            )
        """)
        
        # Shipping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shipping (
                id INTEGER PRIMARY KEY,
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
            logger.info("Seeding SQLite database tables with mock data...")
            
            # Customers
            customers_data = [
                (1, "Alice Johnson", "alice.j@example.com", "Gold"),
                (2, "Bob Smith", "bob.smith@example.com", "Silver"),
                (3, "Charlie Brown", "charlie.b@example.com", "Bronze"),
                (4, "Diana Prince", "diana.p@example.com", "Platinum")
            ]
            cursor.executemany("INSERT INTO customers (id, name, email, loyalty_tier) VALUES (?, ?, ?, ?)", customers_data)
            
            # Products
            products_data = [
                (1, "SuperWidget 3000", 129.99, 45, "Flagship smart home controller with multi-protocol support."),
                (2, "SmartPlug Lite", 24.99, 150, "Energy-monitoring Wi-Fi smart plug."),
                (3, "Acme SoundBar", 199.99, 15, "Dolby Atmos enabled home theater soundbar."),
                (4, "Vision Camera", 89.99, 0, "Outdoor 2K security camera with night vision.")
            ]
            cursor.executemany("INSERT INTO products (id, name, price, stock_quantity, description) VALUES (?, ?, ?, ?, ?)", products_data)
            
            # Orders
            orders_data = [
                (1, 1, 1, "2026-05-10", "Delivered", 1, 129.99),
                (2, 1, 2, "2026-05-12", "Delivered", 2, 49.98),
                (3, 2, 3, "2026-05-20", "Shipped", 1, 199.99),
                (4, 3, 4, "2026-05-21", "Pending", 1, 89.99),
                (5, 4, 1, "2026-05-22", "Cancelled", 1, 129.99)
            ]
            cursor.executemany("INSERT INTO orders (id, customer_id, product_id, order_date, status, quantity, total_amount) VALUES (?, ?, ?, ?, ?, ?, ?)", orders_data)
            
            # Support Tickets
            tickets_data = [
                (1, 1, "How to reset SuperWidget 3000", "Resolved", "Medium", "2026-05-11"),
                (2, 2, "Soundbar connection issue", "In Progress", "High", "2026-05-21"),
                (3, 3, "Vision Camera out of stock", "Open", "Low", "2026-05-22")
            ]
            cursor.executemany("INSERT INTO support_tickets (id, customer_id, subject, status, priority, created_at) VALUES (?, ?, ?, ?, ?, ?)", tickets_data)
            
            # Returns
            returns_data = [
                (1, 5, "Cancelled before shipment", "Approved", 129.99)
            ]
            cursor.executemany("INSERT INTO returns (id, order_id, reason, status, refund_amount) VALUES (?, ?, ?, ?, ?)", returns_data)
            
            # Shipping
            shipping_data = [
                (1, 1, "FedEx", "1Z999AA10123456784", "Delivered"),
                (2, 2, "UPS", "1Z999AA10123456789", "Delivered"),
                (3, 3, "DHL", "DHL8872635412", "In Transit")
            ]
            cursor.executemany("INSERT INTO shipping (id, order_id, carrier, tracking_number, status) VALUES (?, ?, ?, ?, ?)", shipping_data)
            
            conn.commit()
            logger.info("SQLite database seeding complete.")
    except Exception as e:
        logger.error("SQLite initialization error: %s", e)
        conn.rollback()
        raise e
    finally:
        conn.close()

def initialize_oracle() -> None:
    """Initializes the Oracle database with tables and mock seed data if empty."""
    import oracledb
    creds = get_db_credentials()

    try:
        # Try to connect to Oracle 11g database using thin mode
        conn = oracledb.connect(
            user=creds["user"],
            password=creds["password"],
            dsn=get_oracle_dsn(creds)
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

def initialize_database() -> None:
    """Initializes the database (SQLite or Oracle) with schemas and populates mock data."""
    # Also initialize the table metadata store SQLite DB
    initialize_metadata_db()
    
    if settings.DB_TYPE.lower() == "sqlite":
        initialize_sqlite()
    else:
        initialize_oracle()

# Allowed tables scoped per department to satisfy enterprise segregation requirements.
DEPARTMENT_TABLE_SCOPES: Dict[str, Set[str]] = {
    "sales": {"customers", "orders", "products", "shipping"},
    "technical": {"products", "support_tickets"},
    "billing": {"customers", "orders", "returns"},
    "general": {"customers", "orders", "products", "support_tickets", "returns", "shipping"}
}

def register_sandbox_guardrails(engine) -> None:
    """Registers event listeners on the SQLAlchemy engine to block write queries in SQL Sandbox."""
    # Skip event listener registration if engine is a Mock (e.g. during unit tests)
    from unittest.mock import Mock
    if isinstance(engine, Mock):
        return

    from sqlalchemy import event
    import re

    @event.listens_for(engine, "before_cursor_execute")
    def block_write_queries(conn, cursor, statement, parameters, context, executemany):
        statement_upper = statement.strip().upper()
        # List of forbidden SQL commands to sandbox the execution
        forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "RENAME", "TRUNCATE", "REPLACE"]
        
        # Verify if any forbidden keyword is at the start of any command or subquery
        for keyword in forbidden_keywords:
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, statement_upper):
                raise PermissionError(
                    f"Security Alert: Execution of query containing forbidden keyword '{keyword}' is blocked."
                )

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

    # Load dynamic table metadata from SQLite metadata DB and perform semantic matching
    table_desc = get_all_table_metadata()

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
    db = SQLDatabase.from_uri(
        get_db_uri(),
        include_tables=selected_list
    )
    register_sandbox_guardrails(db.engine)
    return db

def test_db_connection() -> bool:
    """Tests connection to the database.

    Returns:
        True if connection succeeds, False otherwise.
    """
    if settings.DB_TYPE.lower() == "sqlite":
        import sqlite3
        try:
            conn = sqlite3.connect(settings.SQLITE_DB_PATH)
            conn.close()
            return True
        except Exception as e:
            logger.error("SQLite connection test failed: %s", e)
            return False

    # Oracle mode
    import oracledb
    creds = get_db_credentials()
    try:
        conn = oracledb.connect(
            user=creds["user"],
            password=creds["password"],
            dsn=get_oracle_dsn(creds)
        )
        conn.close()
        return True
    except Exception as e:
        logger.error("Database connection test failed: %s", e)
        return False
