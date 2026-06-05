import os
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL_NAME = "google_genai:gemini-2.5-flash"

# ----- Tools ----- #

@tool
def add_product_price(product: str) -> float:
    """Look up the price of a product in a catalog."""
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50
    }
    return prices.get(product.lower(), 0.0)

@tool
def apply_discount(prices: list[float], discount_tier: str) -> float:
    """Apply a discount to the total price based on the customer's
    discount tier. Available tiers are 'bronze', 'silver', and 'gold'."""
    total = sum(prices)
    discount_rates = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount = discount_rates.get(discount_tier.lower(), 0.0)
    return total - (total * (discount / 100))

#  ----- Agent Loop --- #
@traceable(name="agent_loop", project="langchain_tool_calling")
def run_agent(question: str):
    tools = [add_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 50)
    messages = [
        SystemMessage(content=(
            "You are a helpful assistant that can call tools to answer questions. "
            "You have access to the following tools: "
            f"{', '.join(tools_dict.keys())}. "
            "Strictly use the tools to answer the question, and do not make up any information. "
            "You Must call add_product_price to get the price of each product, and then call apply_discount to get the final price after discount. "
            "Pass all product prices as a list to apply_discount."
        )),
        HumanMessage(content=question)
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {i} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # نلحق رسالة الموديل في الـ history فوراً
        messages.append(ai_message)

        if not tool_calls:
            print("\nNo tool calls made. ")
            print(ai_message.content)
            return ai_message.content
        
        # هنلف على كل الـ tool calls المتاحة في اللفة الحالية لأن الموديل ممكن يطلب كذا أداة مع بعض
        for tool_call in tool_calls:
            # حل مشكلة الـ AttributeError بإننا نتعامل مع الـ Tool Call كـ Dict أو كـ Object
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")
            else:
                tool_name = tool_call.name
                tool_args = tool_call.args
                tool_call_id = tool_call.id

            print(f"Tool called: {tool_name} with args: {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool {tool_name} not found in tools dictionary.")
            
            # تنفيذ الأداة
            observation = tool_to_use.invoke(tool_args)
            print(f"Observation: {observation}")

            # إرسال النتيجة للموديل مع الـ id الخاص بالاستدعاء ده
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

if __name__ == "__main__":
    question = "What is the total price for a laptop and headphones with a silver discount?"
    run_agent(question)