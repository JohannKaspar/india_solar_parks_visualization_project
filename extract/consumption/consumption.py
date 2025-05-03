import pandas as pd
import json
import re
import geopandas as gpd

STATE_NAME_MAP = {
    "Andaman- Nicobar": "Andaman and Nicobar",
    "Andhra Pradesh": "Andhra Pradesh",
    "Arunachal Pradesh": "Arunachal Pradesh",
    "Assam": "Assam",
    "Bihar": "Bihar",
    "Chandigarh": "Chandigarh",
    "Chhattisgarh": "Chhattisgarh",
    "Daman & Diu": "Daman and Diu",
    "Delhi": "Delhi",
    "Goa": "Goa",
    "Gujarat": "Gujarat",
    "Haryana": "Haryana",
    "Himachal Pradesh": "Himachal Pradesh",
    "Jammu and Kashmir and Ladakh": "Jammu and Kashmir", 
    "Jharkhand": "Jharkhand",
    "Karnataka": "Karnataka",
    "Kerala": "Kerala",
    "Lakshadweep": "Lakshadweep",
    "Madhya Pradesh": "Madhya Pradesh",
    "Maharashtra": "Maharashtra",
    "Manipur": "Manipur",
    "Meghalaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nagaland": "Nagaland",
    "Odisha": "Orissa",
    "Puducherry": "Puducherry",
    "Punjab": "Punjab",
    "Rajasthan": "Rajasthan",
    "Sikkim": "Sikkim",
    "Tamil Nadu": "Tamil Nadu",
    "Telangana": "Telangana",
    "Tripura": "Tripura",
    "Uttar Pradesh": "Uttar Pradesh",
    "Uttarakhand": "Uttaranchal",
    "West Bengal": "West Bengal",
}

def normalize_state_name(state):
    s = state.strip().replace('#', '').replace('*', '').strip()
    if re.search(r'dadra.*nagar.*haveli', s, re.I) or re.search(r'daman.*diu', s, re.I):
        return "Dadra & Nagar Haveli and Daman & Diu"
    if re.match(r'tripura', s, re.I):
        return "Tripura"
    if re.search(r'j.*k.*ladakh', s, re.I):
        return "Jammu and Kashmir and Ladakh"
    return s

def fetch_and_aggregate_cea():
    # Load the JSON file downloaded by curl
    with open("extract/consumption/psp_energy.json", "r") as f:
        data = json.load(f)

    rows = []
    for year, records in data.items():
        for rec in records:
            state = normalize_state_name(rec["State"])
            # Remove region aggregates and keep only states/UTs
            if "Region" in state or "All India" in state or "DVC" in state or "North-Eastern" in state or state.startswith("Others"):
                continue
            # Parse year from fy or Month
            year_val = year.split('-')[1]
            # Parse energy requirement as float
            try:
                energy = float(rec["energy_requirement"])
            except Exception:
                continue
            rows.append({
                "state": state,
                "year": int(year_val),
                "energy_requirement": round(energy, 2)
            })

    df = pd.DataFrame(rows)
    # Aggregate by state and year (sum over months)
    df_grouped = df.groupby(['state', 'year'])['energy_requirement'].sum().reset_index()
    # Save to CSV
    df_grouped.to_csv("extract/consumption/state_energy_consumption.csv", index=False)
    print("Saved to extract/consumption/state_energy_consumption.csv")

if __name__ == "__main__":
    fetch_and_aggregate_cea()
    # Load state boundaries
    gdf = gpd.read_file('extract/consumption/india_telengana.geojson')

    # Load energy consumption
    consumption = pd.read_csv('extract/consumption/state_energy_consumption.csv')

    # Map CSV state names to GeoJSON names
    consumption['state_geo'] = consumption['state'].map(STATE_NAME_MAP)

    # Merge on the mapped name
    merged = gdf.merge(consumption, left_on='NAME_1', right_on='state_geo', how='left')

    # Save one file per year
    for year in merged['year'].dropna().unique():
        merged_year = merged[merged['year'] == year]
        merged_year.to_file(f'extract/consumption/state_energy_consumption_{int(year)}.geojson', driver='GeoJSON')
        print(f"Saved: extract/consumption/state_energy_consumption_{int(year)}.geojson")
