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