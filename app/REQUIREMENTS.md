# Streamlit Dashboard Requirements: Solar Detective

## Overview
This document outlines the requirements for building a Streamlit-based dashboard to map and explore India's solar infrastructure projects. The dashboard is designed for rapid prototyping and hackathon implementation.

---

## 1. Features

### a. Interactive Map
- Display a map of India with markers for each solar project.
- Markers should be color-coded by project status (e.g., planned, realized) or capacity.
- Support for heatmaps or colored areas to show project density or capacity distribution.
- Tooltip on marker hover showing project name, capacity, and key details.

### b. Filterable Table
- Display a table of all solar projects with columns:
  - Name
  - Capacity (MW)
  - State
  - Opening/Commissioning Year
  - Location (lat/lon)
- Table should be sortable and filterable (e.g., by capacity, state).

### c. Timeline/Slider
- Provide a slider or timeline to filter projects by commissioning year or estimated realization date.

### d. Filtering Controls
- UI controls (dropdowns, sliders, checkboxes) to filter projects by:
  - Capacity range
  - State/region
  - Project status/type (planned, realized, etc.)

### e. Project Details Popup
- When a marker is hovered (or clicked, if possible), show a tooltip with:
  - Project name
  - Capacity
  - Developer/Owner
  - Year of commissioning
  - Additional metadata if available

---

## 2. Layout
- Map should be the main focus, occupying the central or upper part of the dashboard.
- Table should be placed below or beside the map.
- Filters and timeline controls can be placed in a sidebar or above the map.
- Responsive layout for different screen sizes (desktop priority).

---

## 3. Interactivity
- All filters and timeline controls should update both the map and the table in real time.
- Hovering over a marker shows a tooltip with project details.
- Table rows and map markers should be linked (optional: highlight marker when table row is hovered/selected).

---

## 4. Technical Notes
- Use Streamlit as the main framework.
- Use `pydeck` for the map visualization (supports markers, heatmaps, tooltips).
- Use `st.dataframe` or `st-aggrid` for the table.
- Use Streamlit widgets for filters and timeline.
- Data can be loaded from a CSV, database, or API (for hackathon, CSV is sufficient).
- Prioritize speed of development and ease of use for hackathon demo.

---

## 5. Stretch Goals (Optional)
- Add export functionality (CSV, image, etc.).
- Add more advanced map layers (e.g., power grid overlays).
- Support for uploading new project data.
- More advanced popups or modal dialogs for project details.

---

## 6. References
- See README.md for project background and data sources. 