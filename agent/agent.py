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

# Configuration
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY


from pydantic import BaseModel, Field, condecimal


class Project(BaseModel):
    name: str = Field(..., description="Official project name or title (e.g. “Yavatmal Solar Park Phase I”)")
    capacity_mw: condecimal(max_digits=8, decimal_places=2) = Field(..., description="Installed capacity in megawatts (numeric, e.g. 150.00)")
    latitude: condecimal(max_digits=9, decimal_places=6) = Field(..., description="Latitude of the project site in decimal degrees (e.g. 20.1548)")
    longitude: condecimal(max_digits=9, decimal_places=6) = Field(..., description="Longitude of the project site in decimal degrees (e.g. 78.1518)")
    year_commissioned: Optional[int] = Field(None, description="Year when the project began commercial operation (4‑digit year, e.g. 2023)")
    commission_date: Optional[date] = Field(None, description="Full commissioning date if known (ISO 8601 date, e.g. “2023-08-15”)")
    type: Literal["Utility", "Rooftop", "Floating", "Hybrid"] = Field(..., description="Project category: one of Utility, Rooftop, Floating, Hybrid")


# class TechnicalDetails(BaseModel):
#     cell_technology: str = Field(..., description="Solar cell material/technology (e.g. “c-Si”, “CdTe”, “Perovskite”)")
#     bifacial: bool = Field(..., description="Boolean flag indicating if panels are bifacial (True/False)")
#     grid_infra_description: str = Field(..., description="Text description of grid interconnection (substation name, voltage level, line distance)")


# class Manufacturer(BaseModel):
#     name: str = Field(..., description="Manufacturer or supplier name (e.g. “First Solar, Inc.”)")
#     component: str = Field(..., description="Component type supplied (e.g. “Panels”, “Inverters”, “Mounting Structure”)")

  
# class OfftakeAgreement(BaseModel):
#     type: Literal["PPA", "Merchant"] = Field(..., description="Agreement type: “PPA” or “Merchant”")
#     counterparty: str = Field(..., description="Name of buyer or offtaker (e.g. utility or corporate offtaker)")
#     start_date: date = Field(..., description="Agreement start date (ISO 8601 date, e.g. “2023-01-01”)")
#     end_date: Optional[date] = Field(None, description="Agreement end date (ISO 8601 date, or null if open‑ended)")
#     price: condecimal(max_digits=10, decimal_places=2) = Field(..., description="Contracted price per MWh (numeric, e.g. 55.00)")
#     currency_code: str = Field(..., description="ISO 4217 currency code for price (e.g. “INR” or “USD”)")
    

# class Financing(BaseModel):
#     amount: condecimal(max_digits=20, decimal_places=2) = Field(..., description="Financing amount (numeric, e.g. 100000000.00)")
#     currency_code: str = Field(..., description="ISO 4217 code of financing currency (e.g. “USD”)")
#     financier: str = Field(..., description="Name of financier or lending institution")
#     date: date = Field(..., description="Date financing was secured (ISO 8601 date)")


# class Entity(BaseModel):
#     name: str = Field(..., description="Entity name (developer, owner or operator)")
#     role: Literal["Developer", "Owner", "Operator"] = Field(..., description="Role of this entity: “Developer”, “Owner” or “Operator”")


# class Source(BaseModel):
#     url: str = Field(..., description="Full URL of the data source (web page, PDF link)")
#     retrieved_at: datetime = Field(..., description="DateTime when the source was accessed (ISO 8601 timestamp)")


# class SourceField(BaseModel):
#     table_name: str = Field(..., description="Name of the source table or document section")
#     column_name: str = Field(..., description="Name of the source column or field")
#     recorded_at: datetime = Field(..., description="Timestamp when this field was extracted (ISO 8601)")


# Schema columns for extraction
schema_columns = {
    "projects_and_entities": {
        "capacity_mw": "Installed capacity in megawatts (numeric, e.g. 150.00)",
        "latitude": "Latitude of the project site in decimal degrees (e.g. 20.1548)",
        "longitude": "Longitude of the project site in decimal degrees (e.g. 78.1518)",
        "year_commissioned": "Year when the project began commercial operation (4‑digit year, e.g. 2023)",
        "commission_date": "Full commissioning date if known (ISO 8601 date, e.g. “2023-08-15”)",
        "type": "Project category: one of Utility, Rooftop, Floating, Hybrid",
        "name": "Entity name (developer, owner or operator)",
        "role": "Role of this entity: “Developer”, “Owner” or “Operator”"
    },
    "technical_details_and_manufacturers": {
        "cell_technology": "Solar cell material/technology (e.g. “c-Si”, “CdTe”, “Perovskite”)",
        "bifacial": "Boolean flag indicating if panels are bifacial (True/False)",
        "grid_infra_description": "Text description of grid interconnection (substation name, voltage level, line distance)",
        "name": "Manufacturer or supplier name (e.g. “First Solar, Inc.”)",
        "component": "Component type supplied (e.g. “Panels”, “Inverters”, “Mounting Structure”)"
    },
    "offtake_agreements_and_financing": {
        "type": "Agreement type: “PPA” or “Merchant”",
        "counterparty": "Name of buyer or offtaker (e.g. utility or corporate offtaker)",
        "start_date": "Agreement start date (ISO 8601 date, e.g. “2023-01-01”)",
        "end_date": "Agreement end date (ISO 8601 date, or null if open‑ended)",
        "price": "Contracted price per MWh (numeric, e.g. 55.00)",
        "offtake_agreement_currency_code": "ISO 4217 currency code for price (e.g. “INR” or “USD”)",
        "amount": "Financing amount (numeric, e.g. 100000000.00)",
        "financing_currency_code": "ISO 4217 code of financing currency (e.g. “USD”)",
        "financier": "Name of financier or lending institution",
        "date": "Date financing was secured (ISO 8601 date)"
    },
    # "sources": {
    #     "url": "Full URL of the data source (web page, PDF link)",
    #     "retrieved_at": "DateTime when the source was accessed (ISO 8601 timestamp)"
    # },
    # "source_fields": {
    #     "table_name": "Name of the source table or document section",
    #     "column_name": "Name of the source column or field",
    #     "recorded_at": "Timestamp when this field was extracted (ISO 8601)"
    # }
}


def add_strings(previous: Optional[List[str]], new: Union[str, List[str]]) -> List[str]:
    """
    Reducer for plain strings (or list of strings).
    - `previous` is the accumulated list so far (or None).
    - `new`     is either one string or a list of strings to append.
    """
    out: List[str] = [] if previous is None else previous.copy()
    if isinstance(new, list):
        out.extend(new)
    else:
        out.append(new)
    return out


def add_dicts(
    previous: Optional[List[Dict[str, Any]]],
    new: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Reducer for dict objects (or list of dicts).
    - `previous` is the accumulated list so far (or None).
    - `new`      is either one dict or a list of dicts to append.
    """
    out: List[Dict[str, Any]] = [] if previous is None else previous.copy()
    if isinstance(new, list):
        out.extend(new)
    else:
        out.append(new)
    return out


def add_tuples(
    previous: Optional[List[tuple]],
    new: Union[tuple, List[tuple]],
) -> List[tuple]:
    """
    Reducer for tuple objects (or list of tuples).
    - `previous` is the accumulated list so far (or None).
    - `new`      is either one tuple or a list of tuples to append.
    """
    out: List[tuple] = [] if previous is None else previous.copy()
    if isinstance(new, list):
        out.extend(new)
    else:
        out.append(new)
    return out


# State definition
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    name: Annotated[list[str], add_strings]
    capacity_mw: Annotated[list[float], add_strings]
    latitude: Annotated[list[float], add_strings]
    longitude: Annotated[list[float], add_strings]
    year_commissioned: Annotated[list[Optional[int]], add_strings]
    commission_date: Annotated[list[Optional[date]], add_strings]
    type_of_solar_panel: Annotated[
        list[str], add_strings
    ]  # e.g., ['Utility', 'Rooftop', 'Floating', 'Hybrid']

    # Entities involved in the project
    entities__name: Annotated[list[str], add_strings]
    entities__role: Annotated[
        list[str], add_strings
    ]  # e.g., ['Developer', 'Owner', 'Operator']

    # Technical details
    cell_technology: Annotated[list[Optional[str]], add_strings]  # e.g., 'c-Si', 'CdTe'
    bifacial: Annotated[list[Optional[bool]], add_strings]
    grid_interconnection_infrastructure: Annotated[list[Optional[str]], add_strings]

    # Manufacturers
    manufacturers__name: Annotated[list[str], add_strings]
    manufacturers__component: Annotated[
        list[str], add_strings
    ]  # e.g., 'Panels', 'Inverter'

    # Offtake agreements
    offtake_agremeent__type: Annotated[
        list[str], add_strings
    ]  # e.g., 'PPA', 'Merchant'
    offtake_agremeent__counterparty: Annotated[list[Optional[str]], add_strings]
    offtake_agremeent__start_date: Annotated[list[Optional[date]], add_strings]
    offtake_agremeent__end_date: Annotated[list[Optional[date]], add_strings]
    offtake_agremeent__price: Annotated[list[Optional[float]], add_strings]
    offtake_agremeent__currency_code: Annotated[
        list[str], add_strings
    ]  # TODO set default to INR

    # Financing
    financing__amount: Annotated[list[Optional[float]], add_strings]
    financing__currency_code: Annotated[list[str], add_strings]
    financing__financier: Annotated[list[Optional[str]], add_strings]
    financing__date: Annotated[list[Optional[date]], add_strings]

    # Sources
    source__url: Annotated[list[str], add_strings]
    source__retrieved_at: Annotated[list[datetime], add_strings]
    source__table_name: Annotated[list[str], add_strings]
    source__column_name: Annotated[list[str], add_strings]
    source__recorded_at: Annotated[list[datetime], add_strings]

    searched_queries: Annotated[list[str], add_strings]
    query: str
    urls: Annotated[list[tuple], add_tuples]
    docs: list[dict]  # Annotated[list[dict], add_dicts]


# Node runnables
@tracer.chain
def input_node(state: State) -> dict:
    return {"query": state["name"]}


@tracer.chain
def get_brave_search_node():
    # wrapper to handle timeouts
    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def search_with_backoff(search, query: str) -> str:
        return search.run(query, num_results=5)

    search = BraveSearch.from_api_key(api_key=settings.BRAVE_SEARCH_API_KEY)

    def brave_search_node(state: State) -> dict:
        if state["query"] == state["name"]:
            # Initial query
            search_results = search_with_backoff(search, state["query"][0])
        else:
            # Use the query from the previous step and limit the search to 5 results
            search_results = search_with_backoff(search, state["query"][0])
        urls = json.loads(search_results)
        urls = [url for url in urls if url not in state["urls"]]
        return {"messages": urls, "searched_queries": state["query"][0], "urls": urls}

    return brave_search_node

@tracer.chain
def get_semantic_url_checker_node():
    
    def semantic_url_checker_node(state: State) -> dict:
        
        return {"valid_urls": state["urls"]}

    return semantic_url_checker_node

@tracer.chain
def download_node(state: State) -> dict:
    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(min=1, max=15),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    # TODO implement PDF support
    def fetch_with_retry(url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://google.com/",
            "Connection": "keep-alive",
        }
        resp = requests.get(url, headers=headers, timeout=2)
        resp.raise_for_status()
        return resp.text

    retrieved_docs = []
    for url in state["urls"]:
        try:
            if "pdf" in url:
                doc = {}
                loader = PDFPlumberLoader(url)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_documents(documents)
                for text in texts:
                    doc["text"] += text.page_content
            else:
                html = fetch_with_retry(url)
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text()  # TODO check if this does not contain any html tags
                # replace multiple newlines and tabs with spaces using regex
                # text = re.sub(r"\s+", " ", text).strip()
                doc = {"text": text}
            doc["url"] = url
            doc["retrieved_at"] = datetime.utcnow().isoformat()
            retrieved_docs.append(doc)
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

    return {"docs": retrieved_docs}

# def build_json_schema(table_name: str, props: Dict[str,str]) -> Dict[str,Any]:
#     """Produce a valid JSON‑Schema dict for OpenAI function calling."""
#     return {
#         "title": f"Extract_{table_name}",
#         "description": f"All fields for the '{table_name}' table.",
#         "type": "object",
#         "properties": {
#             key: {"type": "string", "description": desc}
#             for key, desc in props.items()
#         },
#         # you could pick required = list(props.keys()) or leave empty
#         "required": []
#     }

@tracer.chain
def get_extract_node():
    extract_prompt = (
        "system",
        """
You are an expert data extractor with a strong focus on solar power. 
Your task is to extract data from the given documents and find as much about the solar park as possible. 
Do not hallucinate or make up data—if you are not sure, leave it empty. All solar parks are in India.
Return JSON matching the requested schema and include source URLs or page numbers.

Example output:
{{
    "capacity_mw": "11.40",
    "latitude": "",
    "longitude": "",
    "year_commissioned": "2006",
    "commission_date": "2006-09-01",
    "type": "Utility",
    "name": "S.A.G. Solarstrom",
    "role": "Owner"
}}
""",
    )
    proxy = get_proxy_client("gen-ai-hub")
    llm = ChatOpenAI(proxy_model_name="gpt-4o", proxy_client=proxy, temperature=0)

    def extract_node(state: State) -> dict:
        extracted: Dict[str, Any] = {}

        for table, cols_with_description in schema_columns.items():
            # build a JSON‑Schema for this table
            # schema = build_json_schema(table, cols_with_description)
            missing_cols = 0
            for col in cols_with_description.keys():
                if col not in state:
                    if missing_cols not in state["searched_queries"]:
                        missing_cols += 1
            if missing_cols <= 1:
                continue
        
            for doc in state["docs"]:
                # build prompt
                query_gen_prompt = ChatPromptTemplate.from_messages([
                    extract_prompt,
                    ("user", "{messages}")
                ])
                prompt = {
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"Extract these columns for '{table}': "
                            f"{', '.join(cols_with_description.keys())}\n\n"
                            f"Document text:\n{doc['text']}"
                            f"\n\nUse this schema:\n{json.dumps(cols_with_description, indent=2)}"
                        )
                    }]
                }

                # attach proper function/schema
                query_gen = query_gen_prompt | llm

                # invoke
                print(f"Extracting {table} from {doc['url']}")
                model_output = query_gen.invoke(prompt)

                if model_output == '```json\n[]\n```':
                    continue

                try:
                    model_output = '{' + model_output.content.split("{", 1)[1].rsplit("}")[0] + '}'
                except Exception as e:
                    print(f"Failed to parse JSON: {model_output}")
                    continue
                try:
                    model_output = json.loads(model_output)
                except json.JSONDecodeError as e:
                    system_prompt = (
                        "system",
                        "You are an expert json parser. You fix jsons with syntax errors. Output only the fixed json. Remember to replace all curly quotes with straight quotes. ",
                    )
                    system_prompt = ChatPromptTemplate.from_messages([
                        system_prompt,
                        ("user", "{messages}")
                    ])

                    prompt = {
                        "messages": [{
                            "role": "user",
                            "content": (
                                f"The json shows the error: {e}.\nFix this JSON:\n{model_output}"
                            )
                        }]
                    }
                    fix_json = system_prompt | llm
                    model_output = fix_json.invoke(prompt)
                    model_output = '{' + model_output.content.split("{", 1)[1].rsplit("}")[0] + '}'
                    try:
                        model_output = json.loads(model_output)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {model_output}")
                        continue

                # model_output is a dict of extracted values
                # merge first non‐empty values
                # TODO handle multiple values
                for k, v in model_output.items():
                    if v not in (None, "", []) and k not in extracted:
                        extracted[k] = v
        return extracted

    return extract_node


@tracer.chain
def write_to_db_node(state: State) -> dict:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    for item in state["extracted"]:
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
    return {}


# @tracer.chain
# def get_new_query_node(state: State) -> dict:
#     # TODO eventuell auch related Begriffe zusammenfassen
#     search_terms: dict[str, str] = {
#         "capacity_mw": "installed capacity MW",
#         "year_commissioned": "commission year",
#         "commission_date": "commission date",
#         "type_of_solar_panel": "solar panel type",
#         "entities__name": "project entities names",
#         "entities__role": "roles of project entities",
#         "cell_technology": "cell technology (c-Si, CdTe etc.)",
#         "bifacial": "bifacial solar panels",
#         "grid_interconnection_infrastructure": "grid interconnection infrastructure",
#         "manufacturers__name": "manufacturers of components",
#         "manufacturers__component": "component types by manufacturer",
#         # "offtake_agremeent__type": "offtake agreement type",  # TODO check what this is and if the search terms are good
#         # "offtake_agremeent__counterparty": "offtake agreement counterparty",  # TODO check what this is and if the search terms are good
#         # "offtake_agremeent__start_date": "offtake agreement start date",  # TODO check what this is and if the search terms are good
#         # "offtake_agremeent__end_date": "offtake agreement end date",  # TODO check what this is and if the search terms are good
#         # "offtake_agremeent__price": "offtake agreement price MWh",  # TODO check if the currency is automatically corrected
#         "financing__amount": "financing amount",
#         "financing__currency_code": "financing currency code",
#         "financing__financier": "project financier",
#         "financing__date": "financing date",
#     }

#     for field, description in search_terms.items():
#         values = state.get(field, [])
#         if not values and field not in state["searched_queries"]:
#             park = state["name"][0] if state["name"] else ""
#             query = park + description
#             return {"query": query, "searched_queries": field}
#     return {"query": ""}


@tracer.chain
def should_continue(state: State) -> Literal["search", END]:
    # Check if there are any new URLs to process
    if state["urls"]:
        return "search"
    else:
        return END


# Build and run StateGraph
@tracer.chain
def build_app() -> Any:
    # TODO low priority: add a newpark node to create a new park in the database
    graph = StateGraph(State)
    graph.add_node("input", RunnableLambda(input_node))
    graph.add_node("search", get_brave_search_node())
    graph.add_node("download", RunnableLambda(download_node))
    graph.add_node("extract", get_extract_node())
    # graph.add_node("write_new_query", RunnableLambda(get_new_query_node))
    graph.add_node("write_to_db", RunnableLambda(write_to_db_node))
    graph.add_edge(START, "input")
    graph.add_edge("input", "search")
    graph.add_edge("search", "download")
    graph.add_edge("download", "extract")
    graph.add_edge("extract", END)
    # graph.add_conditional_edges("write_new_query", should_continue)
    return graph.compile()


def run_app_for_all_projects():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    projects = supa.table("projects").select("name").execute().data
    app = build_app()
    for project in projects:
        park = project["name"]
        print(f"Running app for {park}")
        initial_state = {"name": park, "docs": [], "extracted": []}
        app.invoke(initial_state)


# Entry point
if __name__ == "__main__":
    park = os.environ.get("PARK_NAME", "Lohit Solar Park")
    app = build_app()
    initial_state = {"name": park, "docs": [], "extracted": []}
    app.invoke(initial_state)
