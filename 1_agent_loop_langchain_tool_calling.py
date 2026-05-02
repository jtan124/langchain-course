from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"


# --- Tools (LangChain @tool decorator) ---
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"  >>> Tool called: get_product_price with argument: {product}")
    prices = {"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    return prices.get(product, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold"""
    print(
        f"  >>> Tool called: apply_discount with arguments: price={price}, discount_tier={discount_tier}"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_withItools = llm.bind_tools(tools)

    print(f"User question: {question}")
    print("=" * 50)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess of assume any product price. You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price - do Not pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, ask them which tier to use  - do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"--- Iteration {iteration + 1} ---")
        response = llm_withItools.invoke(messages)
        tool_calls = response.tool_calls

        # Return response content if LLM decide no tool calls are needed to answer the question
        if not tool_calls:
            print(f"No more tool calls. Final response: {response.content}")
            return response.content

        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"   [Tool Selected] {tool_name} with arguments {tool_args}")
        tool_to_call = tools_dict.get(tool_name)
        if tool_to_call is None:
            raise ValueError(f"Tool {tool_name} not found in available tools.")

        observation = tool_to_call.invoke(tool_args)
        print(f"   [Observation] {observation}")

        # Append the tool call and observation to the messages for the next iteration
        messages.append(response)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))

    # If no resolution within max iterations, return None
    print("Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
