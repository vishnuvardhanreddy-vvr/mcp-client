"""MCP Client to make use of mcp servers"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

client = MultiServerMCPClient(
    {
        # "math": {
        #     "command": "python",
        #     # Make sure to update to the full absolute path to your math_server.py file
        #     "args": ["./examples/math_server.py"],
        #     "transport": "stdio",
        # },
        "AllTools": {
            # make sure you start your weather server on port 8000
            "url": "http://127.0.0.1:8000/mcp/",
            "transport": "streamable_http",
        }
    }
)

async def main():
    """Entry Point for code"""
    tools = await client.get_tools()

    async def call_model(state: MessagesState):
        """Async model invoke"""
        response = await model.bind_tools(tools).ainvoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()

    weather_response = await graph.ainvoke({"messages": "what is the temperature in kurnool?"})
    print(weather_response["messages"][-1].content)

# Run the async main function
asyncio.run(main())
