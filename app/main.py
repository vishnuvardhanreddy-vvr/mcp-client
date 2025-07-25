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



# import asyncio
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
# from langchain.agents import create_react_agent, create_tool_calling_agent, AgentExecutor
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchain import hub
# from langchain.prompts import PromptTemplate

# load_dotenv()

# async def main():
#     # Wrap client in async context to properly manage sessions
#     client = MultiServerMCPClient({
#     "DBTools": {
#         "url": "http://127.0.0.1:8000/mcp",
#         "transport": "streamable_http",
#     }
# })
#     tools = await client.get_tools()

#     tool_names = [tool.name for tool in tools]

#     # 💬 Custom ReAct prompt with JSON enforcement
#     prompt = PromptTemplate.from_template("""
#     You are an intelligent assistant that can use tools to answer questions.

#     When you decide to use a tool, always follow this format exactly:

#     Action: <tool_name>
#     Action Input: <JSON formatted input, matching the tool's parameters>

#     Begin!

#     Question: {input}
#     {agent_scratchpad}
#     """)

#     # prompt = prompt_template.partial(tools=tools, tool_names=tool_names)

#     # Initialize your LLM
#     llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

#     # Create the ReAct agent
#     # agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

#     agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

#     # Build executor
#     agent_executor = AgentExecutor(
#         agent=agent,
#         tools=tools,
#         verbose=True,
#         handle_parsing_errors=True
#     )

#     # Proper call format for a ReAct agent: messages field
#     result = await agent_executor.ainvoke({
#         "input": "Return me the just the db name which can be helpful to run following query\n"
#         "'Retrieve the distinct names of partners and their products where the delivery mode is 'Online' (case-insensitive) and the cost amount is greater than 1000.'"
#     })

#     print("Agent output:", result["output"])

# if __name__ == "__main__":
#     asyncio.run(main())
