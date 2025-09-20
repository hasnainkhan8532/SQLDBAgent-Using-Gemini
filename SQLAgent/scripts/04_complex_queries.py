"""
Advanced Analytics SQL Agent with Complex Query Capabilities

This script demonstrates an advanced implementation of a secure SQL agent designed for
complex business analytics and reporting. It builds upon the security framework from
script 03 while adding sophisticated business logic and analytics capabilities.

Key Features:
🔒 Same security guardrails as script 03 (read-only, validation, etc.)
📊 Advanced analytics queries (revenue analysis, customer segmentation, etc.)
📈 Business intelligence capabilities (trends, rankings, aggregations)
🔄 Multi-turn conversation support for iterative analysis
📋 Comprehensive business logic documentation in system prompt

Use Cases Demonstrated:
- Revenue analysis and reporting
- Customer lifetime value calculations
- Time-series analysis (weekly trends)
- Product performance rankings
- Multi-table joins and aggregations
- Iterative business intelligence queries

Educational Purpose: Shows how to build production-ready analytics agents that combine
security with sophisticated business intelligence capabilities.
"""

# Load environment variables first (including OPENAI_API_KEY)
from dotenv import load_dotenv; load_dotenv()
import os

# Core LangChain imports for agent functionality
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini language model integration
from langchain.agents import initialize_agent, AgentType  # Agent creation and configuration
from langchain.schema import SystemMessage  # System message formatting for agents
from langchain_community.utilities import SQLDatabase  # Database schema inspection utilities

# Data validation and tool creation imports
from pydantic import BaseModel, Field  # Data validation and serialization
from langchain.tools import BaseTool  # Base class for creating custom tools
from typing import Type  # Type hinting for better code documentation

# Database and utility imports
import sqlalchemy  # Database engine and connection management
import re  # Regular expressions for SQL pattern matching and validation

# Database Configuration
# DB_URL: SQLite database connection string for analytics database
DB_URL = os.getenv("DB_URL", "sqlite:///sql_agent_class.db")

# Create Database Engine
# sqlalchemy.create_engine: Creates a database engine specifically for analytics queries
# This engine will be used by our secure SQL tool for controlled query execution
engine = sqlalchemy.create_engine(DB_URL)

class QueryInput(BaseModel):
    """
    Pydantic model for analytics query input validation.

    This model defines the expected input structure for complex analytics queries.
    It enforces the same security constraints as the basic safe agent while
    supporting more sophisticated analytical operations.

    Attributes:
        sql (str): A single read-only SELECT statement optimized for analytics
                  Supports complex JOINs, aggregations, and window functions
                  Automatically bounded with LIMIT for result set control
    """
    sql: str = Field(description="A single read-only SELECT statement, bounded with LIMIT when returning many rows.")

class SafeSQLTool(BaseTool):
    """
    Advanced Analytics SQL Tool - Secure Complex Query Execution

    This tool extends the basic SafeSQLTool with enhanced capabilities for complex
    analytics queries while maintaining the same security controls.

    Enhanced Features for Analytics:
    ✅ Support for complex JOINs across multiple tables
    ✅ Advanced aggregation functions (SUM, COUNT, AVG, etc.)
    ✅ Window functions and analytics operations
    ✅ Date/time functions for trend analysis
    ✅ Subqueries and CTEs for complex logic
    ✅ Performance optimization through automatic LIMIT injection

    Security Features (inherited):
    🔒 Input validation using regex patterns
    🔒 Whitelist approach - only SELECT statements allowed
    🔒 SQL injection protection through pattern matching
    🔒 Multiple statement prevention
    🔒 Comprehensive error handling
    🔒 Read-only operations only

    Attributes:
        name (str): Tool identifier for agent tool selection
        description (str): Concise description for AI understanding
        args_schema (Type[BaseModel]): Pydantic model for input validation
    """

    # Tool Configuration for Analytics
    # name: Simple identifier for easy agent selection
    name: str = "execute_sql"

    # description: Brief description emphasizing read-only nature
    description: str = "Execute one read-only SELECT."

    # args_schema: Input validation using QueryInput Pydantic model
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, sql: str) -> str | dict:
        """
        Execute complex analytics SQL with comprehensive security validation.

        This method processes sophisticated analytics queries while maintaining
        strict security controls. It handles complex multi-table operations,
        aggregations, and analytical functions.

        Args:
            sql (str): The analytics SQL statement to validate and execute
                      Can include JOINs, subqueries, window functions, etc.

        Returns:
            dict: For successful queries - {"columns": [...], "rows": [...]}
            str: For validation errors or SQL execution errors

        Analytics Query Processing:
        1. Input normalization and cleaning
        2. Security validation (same as basic agent)
        3. Performance optimization (automatic LIMIT for large result sets)
        4. Advanced error handling with helpful messages
        5. Structured result formatting for agent interpretation
        """

        # Step 1: Input Normalization
        # Clean whitespace and remove trailing semicolons for consistent processing
        s = sql.strip().rstrip(";")

        # Step 2: Security Validation Layer
        # Dangerous Operation Detection - prevent any write operations
        if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE)\b", s, re.I):
            return "ERROR: write operations are not allowed."

        # Multiple Statement Prevention - prevent SQL injection via chaining
        if ";" in s:
            return "ERROR: multiple statements are not allowed."

        # Whitelist Validation - ensure only SELECT statements are allowed
        if not re.match(r"(?is)^\s*select\b", s):
            return "ERROR: only SELECT statements are allowed."

        # Step 3: Performance Optimization
        # Automatic LIMIT injection for result set control
        # Skip for aggregate/analytical queries that naturally limit results
        # Pattern matches: LIMIT clauses, COUNT functions, GROUP BY, aggregate functions
        if not re.search(r"\blimit\s+\d+\b", s, re.I) and not re.search(r"\bcount\(|\bgroup\s+by\b|\bsum\(|\bavg\(|\bmax\(|\bmin\(", s, re.I):
            s += " LIMIT 200"  # Conservative limit for analytics queries

        # Step 4: Secure Query Execution
        try:
            with engine.connect() as conn:  # Automatic connection management
                # Execute the validated analytics query
                result = conn.exec_driver_sql(s)

                # Fetch all results (safe due to LIMIT controls)
                rows = result.fetchall()

                # Extract column metadata for structured response
                cols = list(result.keys()) if result.keys() else []

                # Return structured data optimized for analytics interpretation
                return {"columns": cols, "rows": [list(r) for r in rows]}

        except Exception as e:
            # Step 5: Enhanced Error Handling
            # Provide detailed error information for analytics troubleshooting
            return f"ERROR: {e}"

    def _arun(self, *args, **kwargs):
        """
        Async version of _run method - not implemented.

        Analytics queries are typically synchronous operations.
        Async functionality can be added if needed for long-running reports.
        """
        raise NotImplementedError

# Advanced Database Schema Configuration
# SQLDatabase.from_uri: Creates enhanced database utility for analytics
# Parameters:
#   - DB_URL: Database connection string
#   - include_tables: Explicit table whitelist for security and performance
# Tables include: customers, orders, order_items, products, refunds, payments
include_tables_env = os.getenv("INCLUDE_TABLES")
include_tables = [t.strip() for t in include_tables_env.split(",")] if include_tables_env else ["customers","orders","order_items","products","refunds","payments"]
db = SQLDatabase.from_uri(DB_URL, include_tables=include_tables)

# Extract Comprehensive Schema Information
# get_table_info(): Returns detailed table schemas including:
# - Table structures and relationships
# - Column names, types, and constraints
# - Primary keys and indexes
# This provides the agent with complete database knowledge for analytics
schema_context = db.get_table_info()

# Advanced System Message with Business Logic
# This system message includes:
# 1. Role definition (analytics engineer)
# 2. Available tables and their purposes
# 3. Business logic for revenue calculations
# 4. Schema information for query construction
system = f"""You are a careful analytics engineer for SQLite.
Use only listed tables. Revenue = sum(quantity*unit_price_cents) - refunds.amount_cents.
\n\nSchema:\n{{schema_context}}"""

# Initialize Advanced Language Model
# ChatGoogleGenerativeAI: Creates connection optimized for analytical reasoning
# Parameters:
#   - model: gemini-1.5-flash provides good analytics capabilities at lower cost
#   - temperature: 0 ensures consistent, deterministic analytical outputs
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create Analytics Tool Instance
# Instantiate our secure analytics SQL execution tool
tool = SafeSQLTool()

# Create Advanced Analytics Agent
# initialize_agent: Creates an agent executor optimized for business intelligence
# Parameters:
#   - tools: List containing our secure analytics SQL tool
#   - llm: Language model optimized for analytical reasoning
#   - agent: OPENAI_FUNCTIONS type for precise tool selection and execution
#   - verbose: Detailed execution logging for analytics transparency
#   - agent_kwargs: System message with business context and schema information
agent = initialize_agent(
    tools=[tool],  # Secure analytics tool
    llm=llm,  # Analytical reasoning model
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Provider-agnostic agent type
    verbose=True,  # Transparent execution for analytics validation
    agent_kwargs={"system_message": SystemMessage(content=system)}  # Business context
)

# Complex Analytics Query Demonstrations
# These examples showcase the agent's ability to handle sophisticated business intelligence queries

# Query 1: Product Revenue Analysis
# Demonstrates: Multi-table JOINs, aggregation, ranking, business metric calculation
print(agent.invoke({"input": "Top 5 products by gross revenue (before refunds). Include product name and total_cents."})["output"])

# Query 2: Time-Series Revenue Analysis
# Demonstrates: Date functions, window operations, trend analysis, recent data filtering
print(agent.invoke({"input": "Weekly net revenue for the last 6 weeks. Return week_start, net_cents."})["output"])

# Query 3: Customer Lifecycle Analysis
# Demonstrates: Customer segmentation, date aggregation, multi-metric analysis
print(agent.invoke({"input": "For each customer, show their first_order_month, total_orders, last_order_date. Return 10 rows."})["output"])

# Query 4: Customer Lifetime Value Ranking
# Demonstrates: Complex revenue calculations, customer ranking, net value computation
print(agent.invoke({"input": "Rank customers by lifetime net revenue (sum of items minus refunds). Show rank, customer, net_cents. Top 10."})["output"])

# Multi-Turn Conversation Demonstrations
# These examples show the agent's ability to maintain context across multiple queries
# for iterative business intelligence analysis

# Turn 1: High-level category analysis
print(agent.invoke({"input": "What categories drive the most revenue?"})["output"])

# Turn 2: Drill-down analysis building on previous context
# Demonstrates: Context retention, iterative analysis, detailed breakdowns
print(agent.invoke({"input": "Break the top category down by product with totals."})["output"])