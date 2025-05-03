import json
import os
import re
import requests
from datetime import datetime, date
from typing import Annotated, Any, Literal, TypedDict, List, Optional, Union, Dict

from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import BraveSearch
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.tools import tool

from bs4 import BeautifulSoup
from requests import request

from supabase import create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
)

from typing_extensions import TypedDict

from internal.get_tracer import tracer
from internal.config import settings

from agent.agent import download_node

# Configuration
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY



# State definition
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    projects: list[str]
    new_projects: list[str]

# @tool
# @tracer.chain
# def brave_search_tool(query: str) -> str:
#     """
#     A tool that uses the Brave Search API to search for solar parks in India.
#     """
#     # wrapper to handle timeouts
#     @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
#     def search_with_backoff(search, query: str) -> str:
#         return search.run(query)

#     search = BraveSearch.from_api_key(api_key=settings.BRAVE_SEARCH_API_KEY)
#     search_results = search_with_backoff(search, query)
#     data = json.loads(search_results)[:8]
#     urls = [(item.get("title"), item.get("link")) for item in data if (item.get("title"), item.get("link")) not in state["urls"]]
#     docs = download_node({"current_urls": urls})
#     return {"docs": docs}


@tracer.chain
def get_extract_node(proxy_client: Any) -> Any:
    prompt = [
    (
        "system","""
    You are a data extraction agent with a strong focus in solar energy in india. Your task is to find all solar installations in India. We will provide you with a list of solar parks we have already found. Your task is to find new solar parks and only return their names.
    """
    ),]
    query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("user", "{messages}")]
    )
    query_check = query_check_prompt | ChatOpenAI(
        proxy_model_name="gpt-4o", proxy_client=proxy_client, temperature=0
    ).bind_tools([BraveSearch.from_api_key(api_key=settings.BRAVE_SEARCH_API_KEY)], tool_choice="auto")

    @tracer.chain
    def validate_sql_syntax(state: State):
        user_prompt = f'We have already these solar parks: {state["projects"]}. Find new solar parks in India. Only return their names.'
        return {"new_projects": query_check.invoke(user_prompt)}

    return validate_sql_syntax


@tracer.chain
def write_to_db_node(state: State) -> dict:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    for name in state["new_projects"]:
        supa.table("projects").insert(
            {
                "name": name,
            }
        ).execute()
    return {}

# Build and run StateGraph
@tracer.chain
def build_app() -> Any:
    graph = StateGraph(State)
    graph.add_node("extract", get_extract_node())
    graph.add_node("write_to_db", RunnableLambda(write_to_db_node))
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "write_to_db")
    graph.add_edge("write_to_db", END)
    return graph.compile()



# Entry point
if __name__ == "__main__":
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    projects = (
        supa
        .table("projects")
        .select('name')
        .execute()
        .data
    )

    app = build_app()
    print('hello')
    app.invoke({"projects": projects})
