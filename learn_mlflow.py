from typing import Literal

import mlflow

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv
import os


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Enabling tracing for LangGraph (LangChain)
mlflow.langchain.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LangGraph")


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"


llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_weather]
graph = create_agent(llm, tools)

# Invoke the graph
result = graph.invoke({"messages": [{"role": "user", "content": "what is the weather in sf?"}]})