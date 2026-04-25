from langchain_core.messages import HumanMessage,AIMessage ,ToolMessage
def response_printer(response):
    print("#"*100)
    for message in response["messages"]:
        if isinstance(message, AIMessage):
            print(f"AI: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool: {message.content}")
    print("#"*100)