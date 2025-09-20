"""
Simple Agent Demo - Agent Framework Without Tools

This script demonstrates the most basic usage of LangChain's agent framework
without any tools. It creates an agent that can only talk to the LLM for
conversational AI, showing the agent structure before adding tool capabilities.

Educational Purpose:
- Understand the agent framework structure without tool complexity
- See how agents work at their core (just LLM conversation)
- Learn the foundation before adding SQL tools and capabilities
- Compare agent vs direct LLM usage

Key Concepts:
- Agent framework with zero tools
- Conversational agent pattern
- Agent initialization and invocation
- System message configuration for agents
"""

# Load environment variables first (including OPENAI_API_KEY)
from dotenv import load_dotenv; load_dotenv()

# Import LangChain components for agent creation
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini language model integration
from langchain.agents import initialize_agent, AgentType  # Agent framework
from langchain.schema import SystemMessage  # System message configuration
from langchain.tools import BaseTool  # Base class for creating dummy tools
from pydantic import BaseModel, Field  # Data validation for tool inputs
from typing import Type  # Type hinting

class DummyInput(BaseModel):
    """
    Pydantic model for dummy tool input - not actually used.

    This exists only to satisfy the tool framework requirements.
    """
    query: str = Field(description="Any input - this tool does nothing")

class DummyTool(BaseTool):
    """
    Dummy Tool - Does Nothing But Allows Agent Creation

    This tool exists only to satisfy LangChain's requirement that agents
    must have at least one tool. It doesn't actually do anything useful,
    allowing us to demonstrate pure conversational agent behavior.

    Educational Purpose:
    - Shows that agents require tools (even dummy ones)
    - Demonstrates the tool interface without functionality
    - Allows focus on agent conversation patterns
    """

    name: str = "dummy_tool"
    description: str = "A dummy tool that does nothing - used only for agent framework demo"
    args_schema: Type[BaseModel] = DummyInput

    def _run(self, query: str) -> str:
        """
        Dummy tool execution - returns a message explaining it does nothing.

        Args:
            query (str): Any input (ignored)

        Returns:
            str: Message explaining this is a dummy tool
        """
        return "This is a dummy tool that does nothing. I can only provide information through conversation."

    def _arun(self, *args, **kwargs):
        """Async version - not implemented."""
        raise NotImplementedError

def main():
    """
    Main function demonstrating simple agent usage without tools.

    This function shows how to create and use a LangChain agent without any tools:
    1. Initialize the LLM
    2. Create an agent with zero tools
    3. Configure the agent with a system message
    4. Demonstrate agent invocation vs direct LLM usage

    Key difference: Agent framework structure without tool complexity.
    """

    # Initialize the Language Model
    # ChatGoogleGenerativeAI: Creates connection to Gemini models for the agent
    # Parameters:
    #   - model: Specify which Gemini model to use (e.g., gemini-1.5-flash)
    #   - temperature: Controls response randomness (0 = deterministic)
    print("🤖 Initializing language model for agent...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    # Define System Message for Agent
    # This sets the agent's personality and behavior
    system_message = SystemMessage(
        content="""You are a helpful AI assistant specializing in explaining technology concepts.
        You provide clear, concise explanations and are always friendly and professional.
        You have access to one dummy tool, but you should prefer to answer questions directly through conversation."""
    )

    # Create Dummy Tool Instance
    # This tool does nothing but allows the agent to be created
    dummy_tool = DummyTool()

    # Create Agent with Dummy Tool
    # initialize_agent: Creates an agent executor using the agent framework
    # Parameters:
    #   - tools: List with one dummy tool (required by framework)
    #   - llm: Language model for reasoning and responses
    #   - agent: Agent type (ZERO_SHOT_REACT_DESCRIPTION for simple reasoning)
    #   - verbose: Show execution steps for educational purposes
    #   - agent_kwargs: Additional configuration including system message
    print("🎯 Creating agent with dummy tool (conversational focus)...")
    agent = initialize_agent(
        tools=[dummy_tool],          # Dummy tool to satisfy framework requirements
        llm=llm,                     # Language model for conversation
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Simple reasoning agent type
        verbose=True,                # Show agent reasoning process
        agent_kwargs={
            "system_message": system_message  # Set agent personality
        }
    )

    # Agent Invocation - Method 1: Simple question
    # This shows how to use the agent framework for basic conversation
    print("\n💬 Agent conversation (dummy tool available but focus on chat):")
    print("=" * 60)

    # Send a question to the agent
    # agent.invoke(): Processes input through agent framework
    # Even without tools, the agent will use the LLM to respond
    question1 = "What is an AI agent and how does it differ from a chatbot?"
    print(f"Question: {question1}")

    response1 = agent.invoke({"input": question1})
    print(f"Agent Response: {response1['output']}")

    # Agent Invocation - Method 2: Follow-up question
    # Demonstrates that the agent maintains conversation context
    print("\n🔄 Follow-up question:")
    print("=" * 60)

    question2 = "Can you give me a simple example of how agents work?"
    print(f"Question: {question2}")

    response2 = agent.invoke({"input": question2})
    print(f"Agent Response: {response2['output']}")

    # Show Agent Properties
    # The agent object contains useful information about its configuration
    print("\n🔍 Agent configuration:")
    print("=" * 60)
    print(f"Agent type: {type(agent)}")
    print(f"Available tools: {len(agent.tools)} (dummy tool only)")
    print(f"Agent verbose mode: {agent.verbose}")

    # Comparison: Direct LLM vs Agent Framework
    print("\n⚖️  Comparison: Direct LLM vs Agent Framework:")
    print("=" * 60)

    test_question = "Explain the concept of machine learning in one sentence."

    # Direct LLM call
    print("Direct LLM response:")
    direct_response = llm.invoke(test_question)
    print(f"  {direct_response.content}")

    # Agent call
    print("\nAgent framework response:")
    agent_response = agent.invoke({"input": test_question})
    print(f"  {agent_response['output']}")

    print("\n✅ Simple agent demo completed!")
    print("\nKey Takeaways:")
    print("- Agents provide a framework structure even without tools")
    print("- Agent responses go through reasoning steps (shown in verbose mode)")
    print("- Agents can maintain context and handle complex conversations")
    print("- This is the foundation for adding tools like SQL execution")
    print("- Next step: Add tools to give the agent actual capabilities")

# Error Handling Wrapper
# This ensures graceful handling of common issues like missing API keys
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure your .env file contains the required API key")
        print("2. Verify your virtual environment is activated")
        print("3. Check that you've installed requirements: pip install -r requirements.txt")
        print("4. Ensure you have internet connection for API calls")