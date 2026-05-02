from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"


# --- Tools (LangChain @tool decorator) ---
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"  >>> Tool called: get_product_price with argument: {product}")
    prices = {"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    return prices.get(product, 0.0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold"""
    print(
        f"  >>> Tool called: apply_discount with arguments: price={price}, discount_tier={discount_tier}"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product to look up the price for.",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price before discount.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply (bronze, silver, gold).",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# --- Helper: traced Ollama call ---
@traceable(name="Ollama LLM Call", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools_for_llm,
    )


# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"User question: {question}")
    print("=" * 50)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess of assume any product price. You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price - do Not pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, ask them which tier to use  - do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"--- Iteration {iteration + 1} ---")
        response = ollama_chat_traced(messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        # Return response content if LLM decide no tool calls are needed to answer the question
        if not tool_calls:
            print(f"No more tool calls. Final response: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"   [Tool Selected] {tool_name} with arguments {tool_args}")
        tool_to_call = tools_dict.get(tool_name)
        if tool_to_call is None:
            raise ValueError(f"Tool {tool_name} not found in available tools.")

        observation = tool_to_call(**tool_args)
        print(f"   [Observation] {observation}")

        # Append the tool call and observation to the messages for the next iteration
        messages.append(ai_message)
        messages.append({
            "role": "tool",
            "content": str(observation)
        })

    # If no resolution within max iterations, return None
    print("Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
