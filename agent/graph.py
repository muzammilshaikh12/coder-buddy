from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langchain.agents import create_agent

from agent.prompts import *
from agent.states import *
from agent.tools import write_file, read_file, get_current_directory, list_files

_ = load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

planner_llm = llm.with_structured_output(Plan)

architect_llm = llm.with_structured_output(TaskPlan)

planner_chain = (
    PLANNER_PROMPT
    | planner_llm
)

architect_chain = (
    ARCHITECT_PROMPT
    | architect_llm
)

def planner_agent(state: dict) -> dict:
    """
    Converts the user prompt into a structured Plan.
    """

    plan = planner_chain.invoke(
        {
            "user_prompt": state["user_prompt"]
        }
    )

    return {
        "plan": plan
    }


def architect_agent(state: dict) -> dict:
    """
    Converts a Plan into a TaskPlan.
    """

    plan: Plan = state["plan"]

    task_plan = architect_chain.invoke(
        {
            "plan": plan.model_dump_json()
        }
    )

    return {
        "task_plan": task_plan
    }


def coder_agent(state: dict) -> dict:
    """LangGraph tool-using coder agent."""
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    existing_content = read_file.run(current_task.filepath)

    prompt = CODER_PROMPT.invoke(
        {
        "task": current_task.task_description,
        "filepath": current_task.filepath,
        "current_file_content": existing_content,
        }
    )

    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_agent(llm, coder_tools)

    react_agent.invoke(
        {
            "messages": prompt.messages
        }
    )

    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}


graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")
graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)

graph.set_entry_point("planner")
agent = graph.compile()
if __name__ == "__main__":
    result = agent.invoke({"user_prompt": "Build a colourful modern todo app in html css and js"},
                          {"recursion_limit": 100})
    print("Final State:", result)