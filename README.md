# 🌞 Solar Detective: Mapping India's Solar Infrastructure Using Agentic AI

## 📋 Table of Contents
- [Motivation](#-1-motivation--goal-to-achieve)
- [Features](#-2-functionalities--features)
- [Resources](#-3-hints--resources-how-to-build-it)
- [Evaluation](#-4-evaluation-criteria-for-the-prototype)
- [Impact](#-why-this-matters)

## 🎯 1) Motivation / Goal to Achieve

India is scaling up solar energy rapidly, but comprehensive and up-to-date information about existing projects is scattered across government portals, company PDFs, press releases, and industry databases.
This fragmented data makes it difficult for developers, investors, and planners to:
- Track what solar capacity already exists
- Understand where opportunities or infrastructure gaps lie
- Make informed investment or policy decisions

### 🎯 Your Challenge
Build an AI-powered agent that:
1. Scrapes, aggregates, and organizes publicly available data to produce a nationwide map of all commercial solar projects in India
2. Develops a smart dashboard that visualizes this data to offer a clear picture of the country's solar energy landscape, project by project

## ⚙️ 2) Functionalities / Features

### 🤖 AI Agent Capabilities
- Scrape structured and unstructured data from:
  - Government portals (e.g., MNRE, SECI, POSOCO/Grid India)
  - Company investor reports and PDFs
  - News articles and press releases
  - Open satellite imagery / GIS platforms
- Clean, match, and standardize project-level data
- Store in a searchable and filterable database

### 📊 Key Information to Extract Per Project

#### Basic Info
- Capacity (MW)
- Location (lat/long), project images
- Developer / Owner / Operator
- Year of commissioning
- Type: Utility-scale, Rooftop, Floating, Hybrid (e.g., with storage or wind)

#### Technical Details
- Cell technology (c-Si, CdTe, etc.)
- Bifacial vs monofacial
- Grid interconnection infrastructure

#### Business & Policy Details
- Upstream manufacturers
- Offtake agreements (PPA vs merchant market)
- Financing details (if public)
- Historical performance or dispatch data
- Irradiance and grid proximity metrics

### 📱 Dashboard Functions
- Interactive map of all solar projects in India
- Filters by size, type, developer, commissioning year, and location
- Clickable pop-ups with project-level metadata
- Exportable views for policy briefs or investor pitch decks

## 🔧 3) Hints & Resources How to Build It

### 📚 Starter Data Sources
- [MNRE India](https://mnre.gov.in) – Ministry of New & Renewable Energy
- [SECI](https://www.seci.co.in) – Solar Energy Corporation of India
- [POSOCO / Grid India](https://posoco.in/en/) – Real-time dispatch & infrastructure
- [Investor Relations PDFs] from Adani, ReNew, Tata Power, Azure
- [NSEFI](https://www.nsefi.in) – National Solar Energy Federation of India
- [Google Dataset Search](https://datasetsearch.research.google.com) – Aggregated datasets

### 🛠️ Helpful Tools
- **PDF parsing**: pdfplumber, PyMuPDF
- **Scraping**: BeautifulSoup, Selenium, Scrapy
- **Mapping**: Mapbox, Leaflet, Plotly Dash
- **Database**: SQLite or Pandas for prototype; PostgreSQL for scaling
- **Optional**: Use LangChain for document parsing + GPT for summarization

## 🧪 4) Evaluation Criteria for the Prototype

| Criterion | Goal |
|-----------|------|
| Data Coverage | Number and diversity of solar projects mapped from multiple sources |
| Extraction Accuracy | Correct parsing of technical and business details |
| Dashboard Clarity & Usability | Easy to filter, explore, and interpret project data |
| Update & Scalability Readiness | Can the system be reused or updated for other regions or technologies? |
| Impact Potential | Helps real users (planners, investors) make better infrastructure decisions |

## ✨ Why This Matters

India is on track to become one of the world's largest solar power producers, but its success hinges on transparency, coordination, and data-driven planning. With a "Solar Detective" AI agent, we can shine light on the entire energy landscape—empowering smart investments, accelerating new project development, and helping India meet its clean energy goals.
