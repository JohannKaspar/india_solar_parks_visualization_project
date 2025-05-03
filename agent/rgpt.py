from gpt_researcher import GPTResearcher
import asyncio


async def get_report(query: str, report_type: str) -> str:
    researcher = GPTResearcher(query, report_type)
    report = await researcher.run()
    return report

if __name__ == "__main__":
    query = "Find all information about the Amreshwar solar farm in India. Include its location, capacity, and any other relevant details like the owner or supplier."
    report_type = "research_report"

    report = asyncio.run(get_report(query, report_type))
    print(report)


def run_app_for_all_projects():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    projects = supa.table("projects").select("name").execute().data
    app = build_app()
    for project in projects:
        park = project["name"]
        print(f"Running app for {park}")
        initial_state = {"park_name": park, "docs": [], "extracted": []}
        app.invoke(initial_state)



@tracer.chain
def get_extract_node():
    llm = init_llm()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    extract
    def extract_node(state: State) -> dict:
        # TODO loop over all tables and search in the retrieved documents for values to fill its columns

        docs = state["docs"]
        extracted: list[dict] = []

        for table, cols in schema_columns.items():
            data, sources = {}, []
            for doc in docs:
                

            # project_id wird ggf. erst im write‑Step ermittelt
            extracted.append({"table": table, "data": data, "sources": sources})
        return {"extracted": extracted}

    return extract_node