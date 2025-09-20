"""
⚠️ DANGEROUS SQL Agent Demo - ALLOWS ANY SQL INCLUDING DELETE ⚠️

This script demonstrates a DANGEROUS implementation of a SQL agent that executes
ANY SQL command without restrictions, including destructive operations like DELETE, DROP, etc.

This is intentionally risky code used to show what NOT to do in production.

Key Security Issues Demonstrated:
- No input validation or sanitization
- Allows DELETE, DROP, TRUNCATE, ALTER operations
- No user permissions or access controls
- Direct SQL execution without safety checks
- Automatic transaction commits

Educational Purpose: Shows the difference between unrestricted and safe SQL agents.
NEVER use this pattern in production environments!
"""

# ⚠️ DEMO ONLY — allows arbitrary SQL including DELETE
import sqlalchemy  # SQL database engine and connection management
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini language model integration
from langchain.agents import initialize_agent, AgentType  # Agent creation and types
from langchain.tools import BaseTool  # Base class for creating custom tools
from pydantic import BaseModel, Field  # Data validation and serialization
from typing import Type  # Type hinting for better code documentation
from langchain.schema import SystemMessage  # System message formatting for agents
from dotenv import load_dotenv  # Environment variable loading
import os

# Load environment variables from .env file (including OPENAI_API_KEY)
load_dotenv()

# Database Configuration
# DB_URL: SQLite database connection string
# Prefer environment variable if provided
DB_URL = os.getenv("DB_URL", "sqlite:///SQLAgent/sql_agent_class.db")

# Create Database Engine
# sqlalchemy.create_engine: Creates a database engine for connection management
# Parameters:
#   - DB_URL: Connection string for the database
# Returns: Engine object that manages database connections and transactions
engine = sqlalchemy.create_engine(DB_URL)

class SQLInput(BaseModel):
    """
    Pydantic model for SQL tool input validation.

    This model defines the expected input structure for the SQL execution tool.
    It uses Pydantic for automatic validation and serialization.

    Attributes:
        sql (str): The SQL statement to execute - accepts ANY SQL without restrictions
    """
    sql: str = Field(description="Any SQL statement.")  # Field with description for AI understanding

class ExecuteAnySQLTool(BaseTool):
    """
    DANGEROUS Custom Tool - Executes ANY SQL Without Restrictions

    This tool inherits from LangChain's BaseTool and implements unrestricted SQL execution.
    It's designed to show dangerous patterns that should be avoided in production.

    Security Issues:
    - No SQL injection protection
    - No operation type restrictions
    - No user permission checks
    - Automatic transaction commits

    Attributes:
        name (str): Tool identifier for the agent
        description (str): Tool description for the AI to understand its purpose
        args_schema (Type[BaseModel]): Pydantic model for input validation
    """

    # Tool Configuration
    # name: Unique identifier for this tool (used by agent for tool selection)
    name: str = "execute_any_sql"

    # description: Tells the AI what this tool does and when to use it
    description: str = "Executes ANY SQL, including DML/DDL. DEMO ONLY."

    # args_schema: Specifies the input format using the SQLInput Pydantic model
    args_schema: Type[BaseModel] = SQLInput

    def _run(self, sql: str) -> str | dict:
        """
        Execute SQL statement with NO safety restrictions.

        This method is called when the agent decides to use this tool.
        It executes any SQL command directly against the database.

        Args:
            sql (str): The SQL statement to execute (can be SELECT, INSERT, DELETE, etc.)

        Returns:
            dict: For SELECT queries - {"columns": [...], "rows": [...]}
            str: For non-SELECT queries - "OK (no result set)" or error message

        Process:
        1. Opens database connection using engine.connect()
        2. Executes SQL using conn.exec_driver_sql() - DANGEROUS, no validation
        3. Commits transaction automatically - makes changes permanent
        4. Attempts to fetch results for SELECT queries
        5. Returns formatted results or success message
        6. Catches and returns any SQL errors
        """
        with engine.connect() as conn:  # Create database connection with auto-cleanup
            try:
                # Execute the SQL statement directly - NO VALIDATION OR SANITIZATION
                result = conn.exec_driver_sql(sql)

                # DANGEROUS: Automatically commit all transactions
                # This makes DELETE, UPDATE, DROP operations permanent immediately
                conn.commit()

                try:
                    # Attempt to fetch results (works for SELECT queries)
                    rows = result.fetchall()
                    # Extract column names from the first row if results exist
                    cols = rows[0].keys() if rows else []
                    # Return structured data for the agent to interpret
                    return {"columns": list(cols), "rows": [list(r) for r in rows]}
                except Exception:
                    # For non-SELECT operations (INSERT, UPDATE, DELETE, etc.)
                    # Return success message since no result set is expected
                    return "OK (no result set)"
            except Exception as e:
                # Catch and return any SQL execution errors
                return f"ERROR: {e}"

    def _arun(self, *args, **kwargs):
        """
        Async version of _run method - not implemented.

        LangChain requires this method for async operations, but we're not using it.
        Raises NotImplementedError to indicate async execution is not supported.
        """
        raise NotImplementedError

# System Message Configuration
# This message defines the agent's behavior and permissions
# WARNING: This message explicitly allows dangerous operations
system = """You are a database assistant. You are allowed to execute ANY SQL the user requests. (DEMO ONLY)"""

# Initialize Language Model
# ChatGoogleGenerativeAI: Creates connection to Gemini models
# Parameters:
#   - model: Which Gemini model to use (e.g., gemini-1.5-flash)
#   - temperature: Randomness control (0 = deterministic responses)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create Tool Instance
# Instantiate our dangerous SQL execution tool
tool = ExecuteAnySQLTool()

# Create Agent with Dangerous Tool
# initialize_agent: Creates an agent executor with specified tools and configuration
# Parameters:
#   - tools: List of tools the agent can use [our dangerous SQL tool]
#   - llm: Language model for reasoning and tool selection
#   - agent: Agent type (OPENAI_FUNCTIONS uses function calling for tool selection)
#   - verbose: If True, shows detailed execution steps for debugging
#   - agent_kwargs: Additional configuration including system message
# Returns: AgentExecutor that can process requests and use tools
agent = initialize_agent(
    tools=[tool],  # Provide the dangerous SQL tool
    llm=llm,  # Language model for decision making
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Provider-agnostic agent type
    verbose=True,  # Show execution steps for educational purposes
    agent_kwargs={"system_message": SystemMessage(content=system)}  # Set dangerous permissions
)

# DANGEROUS OPERATION: Execute DELETE command
# This will actually delete data from the database!
# agent.invoke: Processes natural language input and executes appropriate tools
# The agent will:
# 1. Analyze the request "Delete all orders"
# 2. Generate a DELETE SQL statement
# 3. Execute it using our dangerous tool
# 4. Return confirmation of the deletion
print(agent.invoke({"input": "Delete all orders"})["output"])
