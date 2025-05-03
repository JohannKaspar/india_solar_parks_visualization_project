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


# Schema columns for extraction
schema_columns = {
    "projects_and_entities": {
        "capacity_mw": "Installed capacity in megawatts (numeric, e.g. 150.00)",
        "latitude": "Latitude of the project site in decimal degrees (e.g. 20.1548)",
        "longitude": "Longitude of the project site in decimal degrees (e.g. 78.1518)",
        "year_commissioned": "Year when the project began commercial operation (4‑digit year, e.g. 2023)",
        "commission_date": "Full commissioning date if known (ISO 8601 date, e.g. “2023-08-15”)",
        "type_of_solar_panel": "Project category: one of Utility, Rooftop, Floating, Hybrid",
        "entities__name": "Entity name (developer, owner or operator)",
        "entities__role": "Role of this entity: “Developer”, “Owner” or “Operator”"
    },
    "technical_details_and_manufacturers": {
        "cell_technology": "Solar cell material/technology (e.g. “c-Si”, “CdTe”, “Perovskite”)",
        "bifacial": "Boolean flag indicating if panels are bifacial (True/False)",
        "grid_interconnection_infrastructure": "Text description of grid interconnection (substation name, voltage level, line distance)",
        "manufacturers__name": "Manufacturer or supplier name (e.g. “First Solar, Inc.”)",
        "manufacturers__component": "Component type supplied (e.g. “Panels”, “Inverters”, “Mounting Structure”)"
    },
    "offtake_agreements_and_financing": {
        "offtake_agremeent__type": "Agreement type: “PPA” or “Merchant”",
        "offtake_agremeent__counterparty": "Name of buyer or offtaker (e.g. utility or corporate offtaker)",
        "offtake_agremeent__start_date": "Agreement start date (ISO 8601 date, e.g. “2023-01-01”)",
        "offtake_agremeent__end_date": "Agreement end date (ISO 8601 date, or null if open‑ended)",
        "offtake_agremeent__price": "Contracted price per MWh (numeric, e.g. 55.00)",
        "offtake_agremeent__currency_code": "ISO 4217 currency code for price (e.g. “INR” or “USD”)",
        "financing__amount": "Financing amount (numeric, e.g. 100000000.00)",
        "financing__currency_code": "ISO 4217 code of financing currency (e.g. “USD”)",
        "financing__financier": "Name of financier or lending institution",
        "financing__date": "Date financing was secured (ISO 8601 date)"
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

    name: str
    capacity_mw: float
    latitude: float
    longitude: float
    year_commissioned: Optional[int]
    commission_date: Optional[date]
    type_of_solar_panel: str # e.g., ['Utility', 'Rooftop', 'Floating', 'Hybrid'

    # Entities involved in the project
    entities__name: str
    entities__role: str  # e.g., ['Developer', 'Owner', 'Operator

    # Technical details
    cell_technology: Optional[str]  # e.g., 'c-Si', 'CdTe
    bifacial: Optional[bool]
    grid_interconnection_infrastructure: Optional[str]

    # Manufacturers
    manufacturers__name: Annotated[list[str], add_strings]
    manufacturers__component: Annotated[
        list[str], add_strings
    ]  # e.g., 'Panels', 'Inverter'

    # Offtake agreements
    offtake_agremeent__type: str  # e.g., 'PPA', 'Merchant
    offtake_agremeent__counterparty: Optional[str]
    offtake_agremeent__start_date: Optional[date]
    offtake_agremeent__end_date: Optional[date]
    offtake_agremeent__price: Optional[float]
    offtake_agremeent__currency_code: str

    # Financing
    financing__amount: Optional[float]
    financing__currency_code: str
    financing__financier: Optional[str]
    financing__date: Optional[date]

    # Sources
    source__url: Annotated[list[str], add_strings]
    source__retrieved_at: Annotated[list[datetime], add_strings]
    source__table_name: Annotated[list[str], add_strings]
    source__column_name: Annotated[list[str], add_strings]
    source__recorded_at: Annotated[list[datetime], add_strings]

    searched_queries: Annotated[list[str], add_strings]
    query: str
    urls: Annotated[list[tuple], add_tuples]
    current_urls: list[str]
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
        return search.run(query)

    search = BraveSearch.from_api_key(api_key=settings.BRAVE_SEARCH_API_KEY)

    def brave_search_node(state: State) -> dict:
        if state["query"] == state["name"]:
            # Initial query
            search_results = search_with_backoff(search, state["query"][0])
        else:
            # Use the query from the previous step and limit the search to 5 results
            search_results = search_with_backoff(search, state["query"][0])
        data = json.loads(search_results)[:8]
        urls = [(item.get("title"), item.get("link")) for item in data if (item.get("title"), item.get("link")) not in state["urls"]]
        return {"searched_queries": state["query"][0], "urls": urls}

    return brave_search_node

@tracer.chain
def get_semantic_url_checker_node():
    proxy = get_proxy_client("gen-ai-hub")
    llm = ChatOpenAI(proxy_model_name="gpt-4o", proxy_client=proxy, temperature=0)
    def semantic_url_checker_node(state: State) -> dict:
        check_prompt = [(
            "system",
            """
            You are an expert data extractor with a strong focus on solar power. 
            I have a list of Website Titles and URLs that I want you to check for their relevance to my solar park.
            Return only the relevant URLs, without any other text. The website titles should not sound too generic (Wikipedia page of generic solar parks) or too specific (e.g. a blog post about a different solar park, that is not the one of interest).
            Do not return any URLs about other solar parks.
            """
        ),
            ("user", "Extract the relevant URLs for: Lohit Solar Park:\nThe URLs and titles are: ('List of Solar Parks in India - Yellow Haze Solar Power', 'https://yellowhaze.in/list-of-solar-parks-in-india/'), ('List of Solar Parks in India - Prakati India', 'https://www.prakati.in/list-of-solar-parks-in-india/'), ('Bhadla Solar Park - Wikipedia', 'https://en.wikipedia.org/wiki/Bhadla_Solar_Park'), ('Mohammed bin Rashid Al Maktoum Solar Park - Wikipedia', 'https://en.wikipedia.org/wiki/Mohammed_bin_Rashid_Al_Maktoum_Solar_Park'), ('Photovoltaic power station - Wikipedia', 'https://en.wikipedia.org/wiki/Photovoltaic_power_station')\n\n"),
            ("assistant", "https://yellowhaze.in/list-of-solar-parks-in-india/,https://prakati.in/list-of-solar-parks-in-india/,")
        ]
        query_gen_prompt = ChatPromptTemplate.from_messages([
                    *check_prompt,
                    ("user", "{messages}")
                ])
        prompt = {
            "messages": [{
                "role": "user",
                "content": (
                    f"Extract the relevant URLs for '{state['name']}':\nThe URLs and titles are: {', '.join([str(url) for url in state['urls']])}\n\n"
                )
            }]
        }

        # attach proper function/schema
        query_gen = query_gen_prompt | llm

        # invoke
        print(f"Checking URLs: {', '.join([str(url) for url in state['urls']])}")
        model_output = query_gen.invoke(prompt)
        print(f"Relevant URLs: {model_output.content}")
        return {"current_urls": model_output.content.split(",")}

    return semantic_url_checker_node

@tracer.chain
def download_node(state: State) -> dict:
    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(min=1, max=15),
        stop=stop_after_attempt(3),
        reraise=True,
    )
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
    for url in state["current_urls"]:
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
                text = re.sub(r"\s+", " ", text).strip()
                doc = {"text": text}
            doc["url"] = url
            doc["retrieved_at"] = datetime.utcnow().isoformat()
            retrieved_docs.append(doc)
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

    return {"docs": retrieved_docs}


@tracer.chain
def get_extract_node():
    extract_prompt = (
        "system",
        """
You are an expert data extractor with a strong focus on solar power. 
Your task is to extract data from the given documents and find as much about the solar park as possible. 
Do not hallucinate or make up data—if you are not sure, leave it empty. Beware there could be multiple solar parks referenced on one website, only take data from the solar park. All solar parks are in India.
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
                if state[col] in ("", [""], "[]", None, [None], [], [[]], [{}]):
                    if col not in state["searched_queries"]:
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
                for k, v in model_output.items():
                    if v not in (None, "", [], [""], "[]", None, [None], [[]], [{}]):
                        try:
                            if state[k] in (None, "", [], [""], "[]", None, [None], [[]], [{}]):
                                extracted[k] = v
                        except KeyError:
                            continue
        return extracted

    return extract_node


def ugly_get_function(state: Dict, key: str) -> dict:
    try:
        if isinstance(state.get(key), list):
            return state.get(key)[0]
        else:
            return state.get(key)
    except (KeyError, IndexError):
        return None

def _normalize_date(val):
    if isinstance(val, date):
        return val.isoformat()
    if isinstance(val, str):
        try:
            # only accept full ISO dates
            return date.fromisoformat(val).isoformat()
        except ValueError:
            return None
    return None


@tracer.chain
def write_to_db_node(state: State) -> dict:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1) PROJECT (manual “upsert”)
    project_data = {
        "name":             ugly_get_function(state, "name"),
        "capacity_mw":      ugly_get_function(state, "capacity_mw"),
        "latitude":         ugly_get_function(state, "latitude"),
        "longitude":        ugly_get_function(state, "longitude"),
        "year_commissioned":ugly_get_function(state, "year_commissioned"),
        "commission_date":  _normalize_date(ugly_get_function(state, "commission_date")),
        "type":             ugly_get_function(state, "type_of_solar_panel"),
    }

    # a) see if it already exists by name
    proj_res = (
        supa
        .table("projects")
        .select("project_id")
        .eq("name", project_data["name"])
        .execute()
    )
    proj_rows = proj_res.data or []

    if proj_rows:
        # b) update existing
        project_id = proj_rows[0]["project_id"]
        supa.table("projects")\
            .update(project_data)\
            .eq("project_id", project_id)\
            .execute()
    else:
        # c) insert new
        ins = (
            supa
            .table("projects")
            .insert(project_data)
            .execute()
        )
        project_id = ins.data[0]["project_id"]

    # 2) ENTITIES → project_entities
    for ent_name, ent_role in zip(ugly_get_function(state, "entities__name"), ugly_get_function(state, "entities__role")):
        # 2a) find or insert entity
        ent_res = (
            supa
            .table("entities")
            .select("entity_id")
            .eq("name", ent_name)
            .eq("role", ent_role)
            .execute()
        )
        ent_rows = ent_res.data or []
        if ent_rows:
            entity_id = ent_rows[0]["entity_id"]
        else:
            ent_ins = (
                supa
                .table("entities")
                .insert({"name": ent_name, "role": ent_role})
                .execute()
            )
            entity_id = ent_ins.data[0]["entity_id"]

        # 2b) link project ↔ entity by checking first, then inserting if missing
        link_res = (
            supa
            .table("project_entities")
            .select("project_id,entity_id")
            .eq("project_id", project_id)
            .eq("entity_id", entity_id)
            .execute()
        )
        if not (link_res.data or []):
            supa.table("project_entities")\
                .insert({
                    "project_id": project_id,
                    "entity_id": entity_id
                }).execute()

    # 3) Technical details
    td = {
        "project_id": project_id,
        "cell_technology": ugly_get_function(state, "cell_technology"),
        "bifacial": ugly_get_function(state, "bifacial"),
        "grid_infra_description": ugly_get_function(state, "grid_interconnection_infrastructure"),
    }
    supa.table("technical_details").upsert(td, on_conflict="project_id").execute()

    # 4) Manufacturers + project_manufacturers

    try:
        for m_name, comp in zip(ugly_get_function(state, "manufacturers__name")):
            man_res = (
                supa
                .table("manufacturers")
                .upsert({"name": m_name}, on_conflict="name")
                .execute()
            )
            man_id = man_res.data[0]["manufacturer_id"]
            supa.table("project_manufacturers").upsert(
                {
                    "project_id": project_id,
                    "manufacturer_id": man_id,
                    "component": comp,
                },
                on_conflict="project_id,manufacturer_id,component"
            ).execute()
    except (TypeError, ValueError):
        # Handle case where manufacturers__name is None or empty
        pass

    # 5) Offtake agreements
    try:
        for t, cp, sd, ed, price, curr in zip(
            ugly_get_function(state, "offtake_agremeent__type"),
            ugly_get_function(state, "offtake_agremeent__counterparty"),
            ugly_get_function(state, "offtake_agremeent__start_date"),
            ugly_get_function(state, "offtake_agremeent__end_date"),
            ugly_get_function(state, "offtake_agremeent__price"),
            ugly_get_function(state, "offtake_agremeent__currency_code"),
        ):
            if not t:
                continue
            off = {
                "project_id": project_id,
                "type": t,
                "counterparty": cp,
                "start_date": sd,
                "end_date": ed,
                "price": price,
                "currency_code": curr,
            }
            supa.table("offtake_agreements").upsert(
                off,
                on_conflict="agreement_id"  # or a natural key if you have one
            ).execute()
    except TypeError:
        # Handle case where offtake_agremeent__type is None or empty
        pass

    # 6) Financing
    try:
        for amt, fcur, fin, dt in zip(
            ugly_get_function(state, "financing__amount"),
            ugly_get_function(state, "financing__currency_code"),
            ugly_get_function(state, "financing__financier"),
            ugly_get_function(state, "financing__date"),
        ):
            if amt is None:
                continue

            # 1) normalize dt into a date object or None
            finance_date = None
            if isinstance(dt, date):
                finance_date = dt
            elif isinstance(dt, int):
                # if it's just a year, default to Jan 1 of that year
                try:
                    finance_date = date(dt, 1, 1)
                except ValueError:
                    finance_date = None
            elif isinstance(dt, str):
                try:
                    finance_date = date.fromisoformat(dt)
                except ValueError:
                    finance_date = None

            record = {
                "project_id":    project_id,
                "amount":        amt,
                "currency_code": fcur,
                "financier":     fin,
                # only include finance_date if we successfully parsed it
                **({"finance_date": finance_date} if finance_date else {}),
            }

            # 2) check if a matching record already exists
            query = supa.table("financing") \
                    .select("finance_id") \
                    .eq("project_id", project_id) \
                    .eq("amount", amt)
            if finance_date:
                query = query.eq("finance_date", finance_date.isoformat())
            existing = query.execute().data or []

            if existing:
                # update the existing row
                supa.table("financing") \
                    .update(record) \
                    .eq("finance_id", existing[0]["finance_id"]) \
                    .execute()
            else:
                # insert a new one
                supa.table("financing") \
                    .insert(record) \
                    .execute()
    except TypeError:
        pass

    state.pop("docs")
    print(state, "\n\n\n")
    return {}


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
    graph.add_node("check_urls", get_semantic_url_checker_node())
    # graph.add_node("write_new_query", RunnableLambda(get_new_query_node))
    graph.add_node("write_to_db", RunnableLambda(write_to_db_node))
    graph.add_edge(START, "input")
    graph.add_edge("input", "search")
    graph.add_edge("search", "check_urls")
    graph.add_edge("check_urls", "download")
    graph.add_edge("download", "extract")
    graph.add_edge("extract", "write_to_db")
    graph.add_edge("write_to_db", END)
    # graph.add_conditional_edges("write_new_query", should_continue)
    return graph.compile()


def make_initial_state(proj: dict) -> State:
    # flatten nested fields into lists
    td = proj.get("technical_details") or {}
    entities = proj.get("project_entities", [])
    manufacturers = proj.get("project_manufacturers", [])
    offtakes = proj.get("offtake_agreements", [])
    financing = proj.get("financing", [])
    sources = proj.get("source_fields", [])  # adjust if you named it differently

    return {
        "messages": [],
        "name": [proj["name"]],
        "capacity_mw": [proj["capacity_mw"]],
        "latitude": [proj["latitude"]],
        "longitude": [proj["longitude"]],
        "year_commissioned": [proj.get("year_commissioned")],
        "commission_date": [proj.get("commission_date")],
        "type_of_solar_panel": [proj.get("type") or ""],

        "entities__name": [
            ent["entities"]["name"] for ent in entities
        ],
        "entities__role": [
            ent["entities"]["role"] for ent in entities
        ],

        "cell_technology": [td.get("cell_technology")],
        "bifacial": [td.get("bifacial")],
        "grid_interconnection_infrastructure": [
            td.get("grid_infra_description")
        ],

        "manufacturers__name": [
            m["manufacturers"]["name"] for m in manufacturers
        ],
        "manufacturers__component": [
            m["component"] for m in manufacturers
        ],

        "offtake_agremeent__type": [
            o["type"] for o in offtakes
        ],
        "offtake_agremeent__counterparty": [
            o.get("counterparty") for o in offtakes
        ],
        "offtake_agremeent__start_date": [
            o.get("start_date") for o in offtakes
        ],
        "offtake_agremeent__end_date": [
            o.get("end_date") for o in offtakes
        ],
        "offtake_agremeent__price": [
            o.get("price") for o in offtakes
        ],
        "offtake_agremeent__currency_code": [
            o.get("currency_code", "") for o in offtakes
        ],

        "financing__amount": [
            f.get("amount") for f in financing
        ],
        "financing__currency_code": [
            f.get("currency_code", "") for f in financing
        ],
        "financing__financier": [
            f.get("financier") for f in financing
        ],
        "financing__date": [
            f.get("finance_date") for f in financing
        ],

        "source__url": [
            s["url"] for s in sources
        ],
        "source__retrieved_at": [
            s["retrieved_at"] for s in sources
        ],
        "source__table_name": [
            s["table_name"] for s in sources
        ],
        "source__column_name": [
            s["column_name"] for s in sources
        ],
        "source__recorded_at": [
            s["recorded_at"] for s in sources
        ],
    }



def run_app_for_all_projects():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    projects = (
        supa
        .table("projects")
        .select(
            """
            *,
            project_manufacturers (
              component,
              manufacturers ( name )
            ),
            project_entities (
              entities ( name, role )
            ),
            offtake_agreements ( * ),
            financing ( * ),
            technical_details ( * )
            """
        )
        .order("capacity_mw", desc=True)
        .execute()
        .data
    )

    app = build_app()
    for i, proj in enumerate(projects[15:]):
        print(f"Processing project {i+1}/{len(projects)}: {proj['name']}")
        initial_state = make_initial_state(proj)
        app.invoke(initial_state)



# Entry point
if __name__ == "__main__":
    run_app_for_all_projects()
