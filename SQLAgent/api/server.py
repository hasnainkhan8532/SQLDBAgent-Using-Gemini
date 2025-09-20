from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Type

import os
import re
import sqlalchemy

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain_community.utilities import SQLDatabase
from langchain.schema import SystemMessage
from pydantic import Field


# Environment configuration
DB_URL = os.getenv("DB_URL", "sqlite:///SQLAgent/sql_agent_class.db")
INCLUDE_TABLES_ENV = os.getenv("INCLUDE_TABLES")
INCLUDE_TABLES = [t.strip() for t in INCLUDE_TABLES_ENV.split(",")] if INCLUDE_TABLES_ENV else None


# Database engine for validated execution
engine = sqlalchemy.create_engine(DB_URL)


class QueryInput(BaseModel):
    sql: str = Field(description="A single read-only SELECT statement, bounded with LIMIT when returning many rows.")


class SafeSQLTool(BaseTool):
    name: str = "execute_sql"
    description: str = "Execute exactly one SELECT statement; DML/DDL is forbidden."
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, sql: str) -> str | dict:
        s = sql.strip().rstrip(";")

        if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE)\b", s, re.I):
            return "ERROR: write operations are not allowed."

        if ";" in s:
            return "ERROR: multiple statements are not allowed."

        if not re.match(r"(?is)^\s*select\b", s):
            return "ERROR: only SELECT statements are allowed."

        if not re.search(r"\blimit\s+\d+\b", s, re.I) and not re.search(r"\bcount\(|\bgroup\s+by\b|\bsum\(|\bavg\(|\bmax\(|\bmin\(", s, re.I):
            s += " LIMIT 200"

        try:
            with engine.connect() as conn:
                result = conn.exec_driver_sql(s)
                rows = result.fetchall()
                cols = list(result.keys()) if result.keys() else []
                return {"columns": cols, "rows": [list(r) for r in rows]}
        except Exception as e:
            return f"ERROR: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError


# Build schema context for prompt
if INCLUDE_TABLES:
    db = SQLDatabase.from_uri(DB_URL, include_tables=INCLUDE_TABLES)
else:
    db = SQLDatabase.from_uri(DB_URL)
SCHEMA_CONTEXT = db.get_table_info()

SYSTEM_MESSAGE = SystemMessage(content=(
    f"You are a careful analytics engineer for SQL. Use only these tables.\n\n{{schema_context}}"
))


def build_agent() -> any:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    tool = SafeSQLTool()
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"system_message": SystemMessage(content=SYSTEM_MESSAGE.content.replace("{schema_context}", SCHEMA_CONTEXT))}
    )
    return agent


app = FastAPI(title="SQL Agent API", version="1.0.0")


class NLQuery(BaseModel):
    input: str


class QueryResponse(BaseModel):
    output: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict:
    return {
        "db_url": DB_URL,
        "include_tables": INCLUDE_TABLES,
        "schema": SCHEMA_CONTEXT,
    }


@app.post("/query", response_model=QueryResponse)
def query(body: NLQuery) -> QueryResponse:
    agent = build_agent()
    result = agent.invoke({"input": body.input})
    return QueryResponse(output=result.get("output", ""))


