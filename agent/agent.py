import json
import os
import re
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict, List

from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import BaseTool, tool
from langchain_community.tools import BraveSearch
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from supabase import create_client
from sqlalchemy import create_engine
from langchain_community.document_loaders import PDFPlumberLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import retry, wait_exponential, stop_after_attempt

from typing_extensions import TypedDict

from internal.get_tracer import tracer
from internal.config import settings

# Configuration
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY


@tracer.chain
def init_llm():
    proxy = get_proxy_client("gen-ai-hub")
    return ChatOpenAI(proxy_model_name="gpt-4o", proxy_client=proxy, temperature=0)


# Schema columns for extraction
schema_columns = {  # TODO add more columns and make it a dict of dicts with explanations
    "projects": [
        "project_id",
        "name",
        "capacity_mw",
        "latitude",
        "longitude",
        "year_commissioned",
        "commission_date",
        "type",
    ],
    "technical_details": ["cell_technology", "bifacial", "grid_infra_description"],
}


# State definition
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    park_name: str
    docs: List[dict]
    extracted: List[dict]


# Node runnables
@tracer.chain
def input_node(state: State) -> dict:
    # Pass through initial park name
    return {"park_name": state["park_name"]}


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def search_with_backoff(search, query: str) -> str:
    return search.run(query)


@tracer.chain
def scrape_node(state: State) -> dict:
    search = BraveSearch.from_api_key(api_key=settings.BRAVE_SEARCH_API_KEY)

    # tracer.start_span('scrape')
    search_results = search.run(f"{state['park_name']} solar park India details")
    print(search_results)
    data = json.loads(search_results)
    urls = [item["link"] for item in data]
    for table, cols in schema_columns.items():
        # tracer.log(f"search for field {col}")
        urls = []
        for col in cols:
            query = f"{state['park_name']} {col} solar park India"
            try:
                result = search_with_backoff(search, query)
                urls += [u for u in result.split("\n") if u.startswith("http")]
            except Exception as e:
                tracer.log(f"BraveSearch failed for '{query}': {e}")
    urls = list(dict.fromkeys(urls))
    docs = []
    for url in urls:
        try:
            loader = (
                PDFPlumberLoader(url)
                if url.lower().endswith(".pdf")
                else WebBaseLoader(url)
            )
            pages = loader.load()
            text = "\n\n".join(p.page_content for p in pages)
            docs.append(
                {
                    "url": url,
                    "text": text,
                    "retrieved_at": datetime.utcnow().isoformat(),
                }
            )
        except Exception as e:
            tracer.log(f"failed to load {url}: {e}")
    # tracer.end_span('scrape')
    return {"docs": docs}


@tracer.chain
def store_node(state: State) -> dict:
    docs = state["docs"]
    # tracer.log(f"storing {len(docs)} documents in memory")
    return {"docs": docs}


@tracer.chain
def extract_node(state: State) -> dict:
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = state["docs"]
    llm = init_llm()
    extracted = []
    for table, cols in schema_columns.items():
        data = {}
        sources = []
        # tracer.start_span(f"extract_{table}")
        for col in cols:
            for doc in docs:
                for chunk in splitter.split_text(doc["text"]):
                    prompt = f"Extract the value for '{col}' from the text, or 'null' if absent.\nText:\n{chunk}"
                    value = llm(prompt).strip()
                    if value.lower() != "null" and value:
                        data[col] = value
                        sources.append(
                            {
                                "url": doc["url"],
                                "column": col,
                                "retrieved_at": doc["retrieved_at"],
                            }
                        )
                        # tracer.log(f"found {col}: {value}")
                        break
                if col in data:
                    break
        # tracer.end_span(f"extract_{table}")
        extracted.append({"table": table, "data": data, "sources": sources})
    return {"extracted": extracted}


@tracer.chain
def write_node(state: State) -> dict:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    for item in state["extracted"]:
        # tracer.start_span('write')
        table = item["table"]
        data = item["data"]
        sources = item["sources"]
        supa.table(table).upsert(data).execute()
        for src in sources:
            res = (
                supa.table("sources")
                .insert({"url": src["url"], "retrieved_at": src["retrieved_at"]})
                .execute()
            )
            sid = res.data[0]["source_id"]
            supa.table("source_fields").insert(
                {
                    "source_id": sid,
                    "project_id": data.get("project_id"),
                    "table_name": table,
                    "column_name": src["column"],
                    "recorded_at": datetime.utcnow().isoformat(),
                }
            ).execute()
        # tracer.end_span('write')
    return {}


@tracer.chain
def newpark_node(state: State) -> dict:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    docs = state["docs"]
    text = " ".join(d["text"] for d in docs)
    for name in set(re.findall(r"[A-Z][a-z]+ Solar Park", text)):
        exists = supa.table("projects").select("name").eq("name", name).execute().data
        if not exists:
            # tracer.log(f"adding new park candidate: {name}")
            supa.table("park_candidates").insert(
                {
                    "name": name,
                    "first_seen": datetime.utcnow().isoformat(),
                    "source_id": None,
                }
            ).execute()
    return {}


# Build and run StateGraph
@tracer.chain
def build_app() -> Any:
    graph = StateGraph(State)
    graph.add_node("input", RunnableLambda(input_node))
    graph.add_node("scrape", RunnableLambda(scrape_node))
    graph.add_node("store", RunnableLambda(store_node))
    graph.add_node("extract", RunnableLambda(extract_node))
    graph.add_node("write", RunnableLambda(write_node))
    graph.add_node("newpark", RunnableLambda(newpark_node))
    graph.add_edge(START, "input")
    graph.add_edge("input", "scrape")
    graph.add_edge("scrape", "store")
    graph.add_edge("store", "extract")
    graph.add_edge("extract", "write")
    graph.add_edge("write", "newpark")
    graph.add_edge("newpark", END)
    return graph.compile()


def run_app_for_all_projects():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    projects = supa.table("projects").select("name").execute().data
    app = build_app()
    for project in projects:
        park = project["name"]
        print(f"Running app for {park}")
        initial_state = {"park_name": park, "docs": [], "extracted": []}
        app.invoke(initial_state)


# Entry point
if __name__ == "__main__":
    park = os.environ.get("PARK_NAME") or input("Enter Solar Park Name: ")
    app = build_app()
    initial_state = {"park_name": park, "docs": [], "extracted": []}
    app.invoke(initial_state)
