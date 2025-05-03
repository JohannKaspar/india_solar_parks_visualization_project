from typing import Annotated, Any, TypedDict

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import BraveSearch
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate



from supabase import create_client

from typing_extensions import TypedDict

from internal.get_tracer import tracer
from internal.config import settings

from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

# Configuration
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY



# State definition
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    projects: list[str]
    new_projects: list[str]


@tracer.chain
def get_extract_node(proxy_client: Any) -> Any:
    prompt =(
        "system","""You are a data extraction agent with a strong focus in solar energy in india. Your task is to find all solar installations in India. We will provide you with a list of solar parks we have already found. Your task is to find new solar parks and only return their names."""
    )
    query_check_prompt = ChatPromptTemplate.from_messages(
        [prompt, ("user", "{messages}")]
    )
    query_check = query_check_prompt | ChatOpenAI(
        proxy_model_name="gpt-4o", proxy_client=proxy_client, temperature=0
    ).bind_tools([BraveSearch.from_api_key(api_key=settings.BRAVE_SEARCH_API_KEY)], tool_choice="auto")

    def extract_node(state: State):
        user_prompt = f'We have already these solar parks: {state["projects"]}. Find new solar parks in India. Only return their names.'
        return {"new_projects": query_check.invoke(user_prompt).content}

    return extract_node


@tracer.chain
def write_to_db_node(state: State) -> dict:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(state["new_projects"])
    if state["new_projects"]:
        print("New projects found:", state["new_projects"])
        for name in state["new_projects"]:
            supa.table("projects").insert(
                {
                    "name": name,
                }
            ).execute()
    else:
        print("No new projects found.")
    return {"new_projects": state["new_projects"]}

# Build and run StateGraph
@tracer.chain
def build_app(proxy) -> Any:
    graph = StateGraph(State)
    graph.add_node("extract", get_extract_node(proxy))
    graph.add_node("write_to_db", RunnableLambda(write_to_db_node))
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "write_to_db")
    graph.add_edge("write_to_db", END)
    return graph.compile()



# Entry point
if __name__ == "__main__":
    proxy = get_proxy_client("gen-ai-hub")
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    projects = (
        supa
        .table("projects")
        .select('name')
        .execute()
        .data
    )

    app = build_app(proxy)
    app.invoke({"projects": projects})
