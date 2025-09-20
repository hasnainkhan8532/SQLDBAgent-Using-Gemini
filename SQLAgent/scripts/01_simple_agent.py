"""
Simple SQL Agent Demo

This script demonstrates the basic usage of LangChain's SQL agent capabilities.
It creates a simple agent that can execute SQL queries against a SQLite database
without any safety restrictions.

Key Components:
- ChatOpenAI: The language model that powers the agent
- SQLDatabase: Wrapper for database connection and operations
- SQLDatabaseToolkit: Pre-built tools for SQL operations
- create_sql_agent: Factory function to create a SQL-capable agent

Safety Note: This agent has NO restrictions and can execute any SQL including
DELETE, DROP, INSERT, etc. It's meant for demonstration purposes only.
"""

# Import necessary LangChain components for SQL agent functionality
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chat model integration
from langchain_community.utilities import SQLDatabase  # Database connection wrapper
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit  # SQL agent tools (updated import)
from langchain_community.agent_toolkits.sql.base import create_sql_agent  # SQL agent factory (updated import)
from langchain.agents import AgentType
from dotenv import load_dotenv; load_dotenv()  # Load environment variables from .env file
import os

# Initialize the Language Model
# ChatGoogleGenerativeAI: Creates a Google Gemini model instance for the agent
# Parameters:
#   - model: Specifies which Gemini model to use (e.g., gemini-1.5-flash)
#   - temperature: Controls randomness (0 = deterministic, 1 = more creative)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create Database Connection
# SQLDatabase.from_uri: Creates a database wrapper from a connection string
# Prefer DB_URL from environment, fallback to demo DB file
db_url = os.getenv("DB_URL", "sqlite:///SQLAgent/sql_agent_class.db")
db = SQLDatabase.from_uri(db_url)

# Create SQL Agent
# create_sql_agent: Factory function that creates a complete SQL-capable agent
# Parameters:
#   - llm: The language model instance to use for reasoning
#   - toolkit: SQLDatabaseToolkit provides pre-built tools for SQL operations
#     - db: Database connection object
#     - llm: Language model for query generation and result interpretation
#   - agent_type: Specifies the agent architecture ("openai-tools" uses function calling)
#   - verbose: If True, prints detailed execution steps for debugging
# Returns: AgentExecutor that can process natural language requests and execute SQL
agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type="tool-calling",
    verbose=True
)

# Execute a Sample Query
# agent.invoke: Executes the agent with a natural language input
# Parameters:
#   - input dict: Contains the natural language request
# Returns: Dict with "output" key containing the agent's response
# Process:
#   1. Agent analyzes the natural language request
#   2. Determines what SQL query to execute
#   3. Executes the query against the database
#   4. Formats and returns the results in natural language
print(agent.invoke({"input": "Show first 5 customers with their regions."})["output"])