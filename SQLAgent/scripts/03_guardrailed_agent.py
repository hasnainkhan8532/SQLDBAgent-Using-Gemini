"""
Safe SQL Agent with Security Guardrails

This script demonstrates a SECURE implementation of a SQL agent that only allows
read-only SELECT operations and implements multiple layers of security controls.

This is the SAFE alternative to the dangerous agent in script 02.

Security Features Implemented:
✅ Input validation using regex patterns
✅ Whitelist approach - only SELECT statements allowed
✅ Automatic LIMIT injection to prevent large result sets
✅ SQL injection protection through pattern matching
✅ Multiple statement prevention
✅ Error handling for SQL execution failures
✅ Read-only operations only - no data modification possible

Educational Purpose: Shows best practices for SQL agent security.
This pattern should be used as a baseline for production implementations.
"""

import re  # Regular expressions for SQL pattern matching and validation
import sqlalchemy  # Database engine and connection management
from pydantic import BaseModel, Field  # Data validation and serialization
from langchain.tools import BaseTool  # Base class for creating custom tools
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini language model integration
from langchain.agents import initialize_agent, AgentType  # Agent creation and configuration
from langchain_community.utilities import SQLDatabase  # Database schema inspection utilities
from langchain.schema import SystemMessage  # System message formatting for agents
from typing import Type  # Type hinting for better code documentation
from dotenv import load_dotenv; load_dotenv()  # Environment variable loading
import os

# Database Configuration
# DB_URL: SQLite database connection string for local development
DB_URL = os.getenv("DB_URL", "sqlite:///sql_agent_class.db")

# Create Database Engine
# sqlalchemy.create_engine: Creates a database engine for connection management
# Used for direct SQL execution with our custom safety checks
engine = sqlalchemy.create_engine(DB_URL)

class QueryInput(BaseModel):
    """
    Pydantic model for safe SQL query input validation.

    This model defines the expected input structure for the safe SQL execution tool.
    It includes clear documentation about what types of queries are allowed.

    Attributes:
        sql (str): A single read-only SELECT statement with automatic LIMIT bounds
    """
    sql: str = Field(description="A single read-only SELECT statement, bounded with LIMIT when returning many rows.")

class SafeSQLTool(BaseTool):
    """
    SECURE SQL Tool - Only Allows Read-Only SELECT Operations

    This tool implements multiple layers of security to prevent dangerous SQL operations.
    It serves as a safe alternative to unrestricted SQL execution tools.

    Security Layers:
    1. Pattern-based validation using regex
    2. Whitelist approach (only SELECT allowed)
    3. Automatic LIMIT injection for result set control
    4. SQL injection pattern detection
    5. Multi-statement prevention
    6. Comprehensive error handling

    Attributes:
        name (str): Tool identifier for agent tool selection
        description (str): Clear description of tool capabilities and restrictions
        args_schema (Type[BaseModel]): Pydantic model for input validation
    """

    # Tool Configuration
    # name: Unique identifier for this tool (simple name for agent understanding)
    name: str = "execute_sql"

    # description: Clear statement of what this tool does and its restrictions
    description: str = "Execute exactly one SELECT statement; DML/DDL is forbidden."

    # args_schema: Input validation using QueryInput Pydantic model
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, sql: str) -> str | dict:
        """
        Execute SQL with comprehensive security validation.

        This method implements multiple security checks before executing any SQL.
        It follows a security-first approach with validation at every step.

        Args:
            sql (str): The SQL statement to validate and execute

        Returns:
            dict: For successful SELECT queries - {"columns": [...], "rows": [...]}
            str: For validation errors or SQL execution errors

        Security Validation Process:
        1. Clean and normalize the input SQL
        2. Check for dangerous SQL operations (INSERT, DELETE, etc.)
        3. Prevent multiple statement execution
        4. Ensure only SELECT statements are allowed
        5. Add automatic LIMIT for result set control
        6. Execute with error handling
        7. Return structured results or error messages
        """

        # Step 1: Clean and Normalize Input
        # Remove leading/trailing whitespace and trailing semicolons
        s = sql.strip().rstrip(";")

        # Step 2: Dangerous Operation Detection
        # Search for write operations using case-insensitive regex
        # This prevents data modification, schema changes, and database destruction
        if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE)\b", s, re.I):
            return "ERROR: write operations are not allowed."

        # Step 3: Multiple Statement Prevention
        # Prevent SQL injection through statement chaining
        # Even after removing trailing semicolon, no internal semicolons allowed
        if ";" in s:
            return "ERROR: multiple statements are not allowed."

        # Step 4: Whitelist Validation
        # Ensure the statement starts with SELECT (case-insensitive)
        # (?is) enables case-insensitive and dot-matches-newline modes
        if not re.match(r"(?is)^\s*select\b", s):
            return "ERROR: only SELECT statements are allowed."

        # Step 5: Automatic LIMIT Injection
        # Prevent accidentally large result sets that could overwhelm the system
        # Skip LIMIT injection for aggregate queries (COUNT, SUM, etc.) or existing LIMIT
        if not re.search(r"\blimit\s+\d+\b", s, re.I) and not re.search(r"\bcount\(|\bgroup\s+by\b|\bsum\(|\bavg\(|\bmax\(|\bmin\(", s, re.I):
            s += " LIMIT 200"  # Default limit of 200 rows

        # Step 6: Safe SQL Execution
        try:
            with engine.connect() as conn:  # Automatic connection cleanup
                # Execute the validated SQL statement
                result = conn.exec_driver_sql(s)

                # Fetch all results (safe because of LIMIT)
                rows = result.fetchall()

                # Extract column names from result metadata
                # Use result.keys() which is more reliable than rows[0].keys()
                cols = list(result.keys()) if result.keys() else []

                # Return structured data for agent processing
                return {"columns": cols, "rows": [list(r) for r in rows]}

        except Exception as e:
            # Step 7: Error Handling
            # Catch and return any SQL execution errors (syntax, missing tables, etc.)
            return f"ERROR: {e}"

    def _arun(self, *args, **kwargs):
        """
        Async version of _run method - not implemented.

        LangChain requires this method definition for potential async operations.
        Since we don't need async functionality, we raise NotImplementedError.
        """
        raise NotImplementedError

# Database Schema Inspection
# SQLDatabase.from_uri: Creates a LangChain database utility for schema inspection
# Parameters:
#   - DB_URL: Database connection string
#   - include_tables: Explicitly list allowed tables for additional security
# Returns: SQLDatabase object with schema inspection capabilities
include_tables_env = os.getenv("INCLUDE_TABLES")
include_tables = [t.strip() for t in include_tables_env.split(",")] if include_tables_env else ["customers","orders","order_items","products","refunds","payments"]
db = SQLDatabase.from_uri(DB_URL, include_tables=include_tables)

# Extract Database Schema Information
# get_table_info(): Returns formatted string containing table schemas
# This provides the agent with knowledge of available tables and columns
schema_context = db.get_table_info()

# System Message Configuration
# This message defines the agent's role and provides database schema context
# The f-string formatting includes the actual table schemas in the message
system = f"You are a careful analytics engineer for SQLite. Use only these tables.\n\n{{schema_context}}"

# Initialize Language Model
# ChatGoogleGenerativeAI: Creates connection to Gemini models
# Parameters:
#   - model: Gemini model selection (e.g., gemini-1.5-flash)
#   - temperature: Controls response randomness (0 = deterministic)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create Safe Tool Instance
# Instantiate our secure SQL execution tool
safe_tool = SafeSQLTool()

# Create Secure Agent
# initialize_agent: Creates an agent executor with safe tools and configuration
# Parameters:
#   - tools: List containing only our safe SQL tool
#   - llm: Language model for reasoning and decision making
#   - agent: Agent type using OpenAI function calling for tool selection
#   - verbose: Show execution steps for educational/debugging purposes
#   - agent_kwargs: Additional configuration including system message with schema
agent = initialize_agent(
    tools=[safe_tool],  # Only provide the safe SQL tool
    llm=llm,  # Language model for decision making
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Provider-agnostic agent type
    verbose=True,  # Show detailed execution for learning
    agent_kwargs={"system_message": SystemMessage(content=system)}  # Include database schema
)

# Test Safe Operations
# First test: Valid read operation that should succeed
print(agent.invoke({"input": "Show 5 customers with their sign-up dates and regions."})["output"])

# Second test: Dangerous operation that should be blocked by security guardrails
# This demonstrates how the agent refuses to execute DELETE operations
print(agent.invoke({"input": "Delete all orders older than July 1, 2025."})["output"])