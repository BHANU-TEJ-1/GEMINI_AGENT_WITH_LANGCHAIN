"""
agent.py

A full tool-using conversational chatbot using:
- Gemini via LangChain
- Custom tools
- Memory (message history)
- Manual agent loop
- System prompt with JSON output format
- Secure API key via .env
"""


import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool

#LOAD GEMINI API KEY
import os
from config import GEMINI_API_KEY

if not GEMINI_API_KEY:
    raise ValueError(" GEMINI_API_KEY not found in config.py")

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

print(" GEMINI_API_KEY Loaded from config.py")


#CONFIGURE GEMINI MODEL

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_output_tokens=512,
    timeout=30,
)


# DEFINING CUSTOM TOOLS

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


@tool
def get_discounted_price(price: float, discount_percent: float) -> float:
    """Calculate final price after discount."""
    if not (0 <= discount_percent <= 100):
        raise ValueError("discount_percent must be between 0 and 100.")
    return price * (1 - discount_percent / 100.0)


tools = [multiply, get_discounted_price]

model_with_tools = model.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}


# SYSTEM PROMPT

system_prompt = SystemMessage(
    content=(
        "You are a helpful AI assistant with access to tools.\n"
        "Rules:\n"
        "1. Use tools when calculation is required.\n"
        "2. Wait for tool outputs before answering.\n"
        "3. The FINAL answer must be in this JSON format:\n"
        "{\n"
        "  \"answer\": \"short clear result\",\n"
        "  \"reasoning\": \"brief explanation\"\n"
        "}\n"
        "Only return valid JSON at the end."
    )
)


# AGENT LOOP

def run_agent(user_input: str, message_history: list) -> list:

    user_msg = HumanMessage(content=user_input)
    message_history.append(user_msg)

    ai_msg = model_with_tools.invoke(message_history)
    message_history.append(ai_msg)

    tool_calls = getattr(ai_msg, "tool_calls", None)

    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool = tool_map.get(tool_name)

            if tool is None:
                tool_output = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    tool_output = tool.invoke(tool_args)
                except Exception as e:
                    tool_output = f"Tool execution error: {e}"

            tool_message = ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_id,
            )

            message_history.append(tool_message)

        final_ai_msg = model.invoke(message_history)
        message_history.append(final_ai_msg)

    else:
        final_ai_msg = ai_msg

    # Extract pure text from Gemini response
    if isinstance(final_ai_msg.content, list):
        ai_text = final_ai_msg.content[0]["text"]
    else:
        ai_text = final_ai_msg.content

    print("\nAI RESPONSE:\n", ai_text)

    return message_history

#  DYNAMIC CHATBOT MODE

if __name__ == "__main__":

    print("\n Gemini Tool-Using Chatbot Started")
    print("Type 'exit' to stop.\n")

    # Auto-check Gemini connection
    try:
        test = model.invoke("Reply with only: OK")
        print(" Gemini API Connection Verified\n")
    except Exception as e:
        print(" Gemini Connection Failed:", e)
        exit(1)

    message_history = [system_prompt]

    while True:
        user_input = input("YOU: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\n Chat ended.")
            break

        message_history = run_agent(user_input, message_history)
