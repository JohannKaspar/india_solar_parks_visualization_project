# Solar Detective: System Architecture Overview

## High-Level Architecture

The Solar Detective system consists of three main components:

1. **Agent (Scraper/ETL)**
   - Periodically scrapes, aggregates, and cleans solar project data from various sources (government portals, PDFs, news, etc.).
   - Processes and standardizes the data.
   - Inserts or updates project data in the central database.

2. **Database (PostgreSQL)**
   - Stores all project, technical, business, and source metadata using a normalized schema (see below).
   - Supports geospatial queries and efficient filtering for the dashboard.
   - Acts as the single source of truth for all solar project data.

3. **Frontend Dashboard (Streamlit App)**
   - Connects to the database to fetch and visualize project data.
   - Displays an interactive map, filterable table, and timeline for exploring solar projects.
   - Provides filtering, searching, and data export capabilities for end users (investors, planners, etc.).

### Data Flow
- The **Agent** scrapes and processes data, then writes it to the **Database**.
- The **Streamlit App** reads from the **Database** (using SQL queries) to display up-to-date information.
- Optionally, the app can support live updates if the agent runs continuously.

---

# Database schema

-- Haupttabelle: Solar‑Projekte
CREATE TABLE projects (
  project_id           SERIAL       PRIMARY KEY,
  name                 TEXT         NOT NULL,
  capacity_mw          DECIMAL(8,2) NOT NULL,
  latitude             DECIMAL(9,6) NOT NULL,
  longitude            DECIMAL(9,6) NOT NULL,
  year_commissioned    SMALLINT     NULL,
  commission_date      DATE         NULL,       -- optionales vollständiges Inbetriebnahmedatum
  type                 VARCHAR(20)  NOT NULL    -- ENUM: 'Utility','Rooftop','Floating','Hybrid'
);
 
-- Bilder zu Projekten (Speicherung als URLs/Referenzen auf externen Object Storage)
CREATE TABLE project_images (
  image_id    SERIAL PRIMARY KEY,
  project_id  INT    REFERENCES projects(project_id) ON DELETE CASCADE,
  image_url   TEXT   NOT NULL,               -- Pfad/URL im Object Storage oder CDN
  caption     TEXT   NULL
);
 
-- Entwickler / Betreiber / Eigentümer
CREATE TABLE entities (
  entity_id   SERIAL PRIMARY KEY,
  name        TEXT    NOT NULL,
  role        VARCHAR(20) NOT NULL         -- 'Developer','Owner','Operator'
);
 
CREATE TABLE project_entities (
  project_id INT REFERENCES projects(project_id) ON DELETE CASCADE,
  entity_id  INT REFERENCES entities(entity_id) ON DELETE CASCADE,
  PRIMARY KEY(project_id, entity_id)
);
 
-- Technische Details
CREATE TABLE technical_details (
  project_id              INT    PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
  cell_technology         VARCHAR(10),            -- z.B. 'c-Si','CdTe'
  bifacial                BOOLEAN,
  grid_infra_description  TEXT
);
 
-- Upstream‑Hersteller (Many‑to‑Many)
CREATE TABLE manufacturers (
  manufacturer_id SERIAL PRIMARY KEY,
  name            TEXT    NOT NULL
);
 
CREATE TABLE project_manufacturers (
  project_id      INT REFERENCES projects(project_id) ON DELETE CASCADE,
  manufacturer_id INT REFERENCES manufacturers(manufacturer_id) ON DELETE CASCADE,
  component       VARCHAR(50),               -- z.B. 'Panels','Inverter'
  PRIMARY KEY(project_id, manufacturer_id, component)
);
 
-- Offtake‑Verträge mit Währungsangabe
CREATE TABLE offtake_agreements (
  agreement_id    SERIAL PRIMARY KEY,
  project_id      INT REFERENCES projects(project_id) ON DELETE CASCADE,
  type            VARCHAR(20) NOT NULL,      -- 'PPA','Merchant'
  counterparty    TEXT        NULL,
  start_date      DATE        NULL,
  end_date        DATE        NULL,
  price           DECIMAL(12,4) NULL,       -- Preis pro MWh
  currency_code   CHAR(3)      NOT NULL DEFAULT 'USD'  -- ISO 4217 Währungscode, default USD
);
 
-- Finanzierung (optional verschiedene Währungen)
CREATE TABLE financing (
  finance_id    SERIAL PRIMARY KEY,
  project_id    INT REFERENCES projects(project_id) ON DELETE CASCADE,
  amount        DECIMAL(14,2),             -- Betrag
  currency_code CHAR(3)      NOT NULL DEFAULT 'USD',  -- ISO 4217 Währungscode
  financier     TEXT,
  finance_date  DATE
);
 
-- Historische Ertrags‑/Dispatch‑Daten (Time‑series)
CREATE TABLE performance_data (
  perf_id            SERIAL PRIMARY KEY,
  project_id         INT REFERENCES projects(project_id) ON DELETE CASCADE,
  timestamp          TIMESTAMP    NOT NULL,
  generation_mwh     DECIMAL(10,3),
  grid_injection_mwh DECIMAL(10,3)
);
 
-- Strahlungs‑ und Netz‑Nähe‑Metriken
CREATE TABLE site_metrics (
  metric_id           SERIAL PRIMARY KEY,
  project_id          INT    REFERENCES projects(project_id) ON DELETE CASCADE,
  metric_date         DATE   NOT NULL,
  irradiance_ghi      DECIMAL(8,3),            -- Global Horizontal Irradiance
  irradiance_dni      DECIMAL(8,3),            -- Direct Normal Irradiance
  distance_to_grid_km DECIMAL(6,3)
);
 
-- Metadaten aller Quellen
CREATE TABLE sources (
  source_id    SERIAL       PRIMARY KEY,
  url          TEXT         NOT NULL,
  retrieved_at TIMESTAMP    NOT NULL DEFAULT now()
);
 
-- Verknüpft Quelle ↔ Projekt ↔ Tabelle/Spalte
CREATE TABLE source_fields (
  source_field_id SERIAL     PRIMARY KEY,
  source_id       INT        NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
  project_id      INT        NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
  table_name      TEXT       NOT NULL,    -- z.B. 'projects' oder 'technical_details'
  column_name     TEXT       NOT NULL,    -- z.B. 'capacity_mw'
  recorded_at     TIMESTAMP  NOT NULL DEFAULT now(),
  UNIQUE(source_id, project_id, table_name, column_name)
);
 