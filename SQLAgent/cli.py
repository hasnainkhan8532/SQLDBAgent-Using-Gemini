import typer
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="CLI client for SQL Agent API")


def get_base_url() -> str:
    return os.getenv("SQL_AGENT_API", "http://127.0.0.1:8000")


@app.command()
def health() -> None:
    r = requests.get(f"{get_base_url()}/health", timeout=30)
    typer.echo(r.json())


@app.command()
def schema() -> None:
    r = requests.get(f"{get_base_url()}/schema", timeout=60)
    typer.echo(r.json())


@app.command()
def ask(question: str) -> None:
    r = requests.post(f"{get_base_url()}/query", json={"input": question}, timeout=120)
    r.raise_for_status()
    typer.echo(r.json().get("output", ""))


if __name__ == "__main__":
    app()


