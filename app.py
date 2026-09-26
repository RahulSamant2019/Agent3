import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Agent Calculator (c) RMS",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 Agent Based Calculator- Designed by Prof. Rahul M. Samant")
st.caption("A teaching demo: LLM → tool selection → deterministic computation → final answer")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Create a key at https://console.groq.com/keys",
    )

    model = st.selectbox(
        "Groq model",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ],
        index=0,
    )

    show_trace = st.checkbox(
        "Show agent/tool trace",
        value=True,
    )

    st.divider()

    st.markdown("### 🎓 What students learn")
    st.markdown(
        """
        1. **LLM** understands the request.
        2. **Agent** decides which tool to use.
        3. **Tool** performs exact arithmetic.
        4. **Observation** is returned to the agent.
        5. **Agent** produces the final answer.
        """
    )

    st.divider()

    st.markdown("### 🛠 Available tools")
    st.code(
        "add(a, b)\n"
        "subtract(a, b)\n"
        "multiply(a, b)\n"
        "divide(a, b)\n"
        "power(a, b)",
	"factorial(a)",
	"is_even(a)",
	"is_prime(a)",
	"is_leap_year(a)",
        language="text",
    )


# ---------------------------------------------------------
# DETERMINISTIC CALCULATOR TOOLS
# ---------------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Returns an error if b is zero."""
    if b == 0:
        return "Error: division by zero is not allowed."
    return a / b


@tool
def power(a: float, b: float) -> float:
    """Raise a to the power b."""
    return a ** b

@tool
def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact

@tool
def is_even(n):
    if n % 2 == 0:
        return "Even"
    else:
        return "Odd"

@tool 
def is_prime(n):
    if n < 2:
        return False

    for i in range(2, n):
        if n % i == 0:
            return False

    return True
@tool
def is_leap_year(year):
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False




TOOLS = [add, subtract, multiply, divide, power, factorial,is_even,is_prime,is_leap_year]


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "trace" not in st.session_state:
    st.session_state.trace = []


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------

def content_to_text(content):
    """Convert LangChain message content to readable text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def make_agent():
    llm = ChatGroq(
        model=model,
        temperature=0,
        api_key=api_key,
    )

    system_prompt = """
You are an educational calculator agent.

Your job is to understand natural-language arithmetic questions and use
the supplied calculator tools for all numerical computation.

Do NOT perform arithmetic mentally when a calculator tool can do it.

For multi-step calculations:
1. Identify the required operations.
2. Call the appropriate tool.
3. Use the returned result in the next operation.
4. Give the user a concise final answer.

Explain the sequence of operations when it helps a student understand
how the agent worked.
"""

    return create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=system_prompt,
    )


# ---------------------------------------------------------
# EXAMPLE QUESTIONS
# ---------------------------------------------------------

st.subheader("Try these examples")

examples = [
    "What is 25 + 17?",
    "Multiply 25 by 17.",
    "Add 25 and 17, then multiply the result by 3.",
    "What is 144 divided by 12?",
    "Calculate 2 raised to the power 10.",
]

cols = st.columns(len(examples))

for i, example in enumerate(examples):
    if cols[i].button(example, key=f"example_{i}"):
        st.session_state["pending_question"] = example
        st.rerun()


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

question = st.chat_input("Ask the calculator agent...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


# ---------------------------------------------------------
# PROCESS REQUEST
# ---------------------------------------------------------

if question:

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        agent = make_agent()

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        messages = result.get("messages", [])

        tool_events = []

        # Collect tool calls and tool results from LangGraph messages.
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            content = getattr(msg, "content", "")

            if msg_type == "ai":
                tool_calls = getattr(msg, "tool_calls", None)

                if tool_calls:
                    for call in tool_calls:
                        tool_events.append(
                            {
                                "type": "call",
                                "name": call.get("name", "unknown"),
                                "args": call.get("args", {}),
                            }
                        )

            elif msg_type == "tool":
                tool_events.append(
                    {
                        "type": "result",
                        "name": getattr(msg, "name", "tool"),
                        "result": content_to_text(content),
                    }
                )

        # Last AI message is normally the final response.
        answer = "No final answer was returned."

        for msg in reversed(messages):
            if getattr(msg, "type", "") == "ai":
                text = content_to_text(getattr(msg, "content", ""))
                if text.strip():
                    answer = text
                    break

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        # -------------------------------------------------
        # EDUCATIONAL TRACE
        # -------------------------------------------------

        if show_trace and tool_events:

            st.divider()
            st.subheader("🔎 Agent Execution Trace")

            st.info(
                "The LLM did not directly calculate the answer. "
                "It selected calculator tools, which performed the arithmetic."
            )

            step = 1

            for event in tool_events:

                if event["type"] == "call":
                    st.markdown(
                        f"**Step {step} — Tool selected:** "
                        f"`{event['name']}`"
                    )

                    st.code(
                        str(event["args"]),
                        language="python",
                    )

                elif event["type"] == "result":
                    st.markdown(
                        f"**Step {step} — Tool result:** "
                        f"`{event['result']}`"
                    )

                step += 1


    except Exception as e:
        st.error("The agent could not complete the request.")
        st.exception(e)


# ---------------------------------------------------------
# RESET
# ---------------------------------------------------------

st.divider()

if st.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    st.session_state.trace = []
    st.rerun()


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.caption(
    "Agentic AI Demonstration by Prof. Rahul M. Samant"
)
