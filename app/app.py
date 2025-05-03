import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import geopandas as gpd
import glob
import re
import numpy as np

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
DEFAULT_MAP_VIEW = {
    'latitude': 22.5937,  # Center of India
    'longitude': 78.9629,
    'zoom': 4,
    'pitch': 0
}

# --- Supabase Data Loading ---
def load_data_from_supabase():
    # Set your Supabase credentials as environment variables
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        st.error("Please set SUPABASE_URL and SUPABASE_KEY as environment variables.")
        st.stop()
    supabase: Client = create_client(supabase_url, supabase_key)
    response = supabase.table('projects').select('*').execute()
    data = response.data
    df = pd.DataFrame(data)
    # Add missing columns for compatibility
    for col in ['state', 'status', 'commission_date']:
        if col not in df.columns:
            df[col] = None
    # Fill numeric columns with 0 and non-numeric columns with "N/A"
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[df.columns.difference(numeric_cols)] = df[df.columns.difference(numeric_cols)].fillna("N/A")

    # Join with entities to get developer, owner, and supplier information
    entities_response = supabase.table('entities').select('*').execute()
    entities_data = entities_response.data
    entities_df = pd.DataFrame(entities_data)
    project_entities_response = supabase.table('project_entities').select('*').execute()
    project_entities_data = project_entities_response.data
    project_entities_df = pd.DataFrame(project_entities_data)
    # Merge entities with project_entities
    project_entities_df = project_entities_df.merge(entities_df, left_on='entity_id', right_on='entity_id', how='left')
    # Group by project_id and role to get lists of developers, owners, and suppliers
    developers = project_entities_df[project_entities_df['role'] == 'Developer'].groupby('project_id')['name'].apply(list).reset_index(name='developer')
    owners = project_entities_df[project_entities_df['role'] == 'Owner'].groupby('project_id')['name'].apply(list).reset_index(name='owner')
    suppliers = project_entities_df[project_entities_df['role'] == 'Supplier'].groupby('project_id')['name'].apply(list).reset_index(name='supplier')
    # Convert lists to strings
    developers['developer'] = developers['developer'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    owners['owner'] = owners['owner'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    suppliers['supplier'] = suppliers['supplier'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    # Merge with main df
    df = df.merge(developers, on='project_id', how='left')
    df = df.merge(owners, on='project_id', how='left')
    df = df.merge(suppliers, on='project_id', how='left')
    # Fill NaN with empty strings
    df['developer'] = df['developer'].fillna('')
    df['owner'] = df['owner'].fillna('')
    df['supplier'] = df['supplier'].fillna('')

    return df

# --- Title and Subtitle ---
def header_section():
    st.title("🌞 Solar Detective: India's Solar Project Dashboard")
    st.write("Explore, filter, and visualize India's solar infrastructure projects with interactive maps and analytics.")

# --- Sidebar Filters ---
def sidebar(df, year_min, year_max, min_capacity, max_capacity):
    st.sidebar.header('Filter Projects')
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        reset = st.button('Reset Filters')
    with col_b:
        reset_map = st.button('Reset Map')
    # State filter
    states = sorted(df['state'].dropna().unique().tolist())
    if 'state' not in st.session_state or reset:
        st.session_state['state'] = []
    selected_state = st.sidebar.multiselect('State', states, key='state')
    # Project type filter
    types = sorted(df['type'].dropna().unique().tolist())
    if 'type' not in st.session_state or reset:
        st.session_state['type'] = []
    selected_type = st.sidebar.multiselect('Project Type', types, key='type')
    # Status filter
    statuses = sorted(df['status'].dropna().unique().tolist())
    if 'status' not in st.session_state or reset:
        st.session_state['status'] = []
    selected_status = st.sidebar.multiselect('Status', statuses, key='status')
    # Developer filter
    developers = sorted(df['developer'].dropna().unique().tolist())
    if 'developer' not in st.session_state or reset:
        st.session_state['developer'] = []
    selected_developer = st.sidebar.multiselect('Developer', developers, key='developer')
    # Owner filter
    owners = sorted(df['owner'].dropna().unique().tolist())
    if 'owner' not in st.session_state or reset:
        st.session_state['owner'] = []
    selected_owner = st.sidebar.multiselect('Owner', owners, key='owner')
    # Supplier filter
    suppliers = sorted(df['supplier'].dropna().unique().tolist())
    if 'supplier' not in st.session_state or reset:
        st.session_state['supplier'] = []
    selected_supplier = st.sidebar.multiselect('Supplier', suppliers, key='supplier')
    # Capacity range filter
    if 'capacity' not in st.session_state or reset:
        st.session_state['capacity'] = (min_capacity, max_capacity)
    capacity_range = st.sidebar.slider('Capacity (MW)', min_capacity, max_capacity, key='capacity')
    # Year filter (for resetting)
    if 'year_slider' not in st.session_state or reset:
        st.session_state['year_slider'] = year_max

    st.sidebar.markdown("---")  # Add a divider line
    st.sidebar.markdown("### Map Settings")  # Add a header for map settings

    # Dot size settings
    size_factor = st.sidebar.slider('Marker Size', min_value=1, max_value=20, value=10, step=1)
    use_capacity_size = st.sidebar.checkbox('Scale marker size by project capacity', value=False)
    
    # Heatmap selection
    heatmap_option = st.sidebar.selectbox('Heatmap', ['No Heatmap', 'Project Heatmap', 'Solar Resource (GHI)', 'State Energy Consumption'], key='heatmap_option')
    return selected_state, selected_type, selected_status, selected_developer, selected_owner, selected_supplier, capacity_range, reset, reset_map, heatmap_option, size_factor, use_capacity_size

# --- Year Slider ---
def year_slider_section(df, year_min, year_max):
    # Filter out rows where year_commissioned is 0 or NaN
    df_filtered = df[df['year_commissioned'].notna() & (df['year_commissioned'] != 0)]
    year_min = int(df_filtered['year_commissioned'].min())
    year_max = int(df_filtered['year_commissioned'].max())
    st.write('### Filter by Commissioned Year')
    selected_year = st.slider('Show projects commissioned up to year:', year_min, year_max, key='year_slider')
    return selected_year

# --- Map Section ---
def map_section(df_filtered, map_view_state, heatmap_option, size_factor, use_capacity_size, selected_year):
    layers = []
    if heatmap_option == 'Solar Resource (GHI)':
        # Load GHI data
        ghi_df = pd.read_csv('data/extracted_ghi_latlon.csv')
        # Remove NaNs
        ghi_df = ghi_df.dropna(subset=['ghi'])
        # Normalize GHI for color mapping (3.0 to 6.0)
        ghi_min, ghi_max = 3.0, 6.0
        def ghi_to_color(ghi):
            t = (ghi - ghi_min) / (ghi_max - ghi_min)
            r = int(255 * (1-t) + 153 * t)
            g = int(255 * (1-t) + 0 * t)
            b = int(153 * (1-t) + 0 * t)
            return [r, g, b, 180]
        ghi_df['color'] = ghi_df['ghi'].apply(ghi_to_color)
        ghi_layer = pdk.Layer(
            'ScatterplotLayer',
            data=ghi_df,
            get_position='[lon, lat]',
            get_radius=10000,
            get_fill_color='color',
            pickable=False,
            opacity=0.5,
        )
        layers.append(ghi_layer)
    elif heatmap_option == 'State Energy Consumption':
        # List all available GeoJSON files for years
        geojson_files = glob.glob('data/state_energy_consumption_*.geojson')
        available_years = []
        for f in geojson_files:
            m = re.search(r'(\d{4})', f)
            if m:
                available_years.append(int(m.group(1)))
        if not available_years:
            st.warning("No state energy consumption data available.")
            return
        closest_year = get_closest_year(selected_year, available_years)
        geojson_path = f'data/state_energy_consumption_{closest_year}.geojson'
        gdf_year = gpd.read_file(geojson_path)
        # Normalize energy_requirement for color mapping
        energy_min = gdf_year['energy_requirement'].min()
        energy_max = gdf_year['energy_requirement'].max()
        def energy_to_color(energy):
            t = (energy - energy_min) / (energy_max - energy_min) if energy_max > energy_min else 0
            r = int(255 * t + 255 * (1-t))
            g = int(255 * (1-t))
            b = int(0)
            return [r, g, b, 180]
        gdf_year['fill_color'] = gdf_year['energy_requirement'].apply(energy_to_color)
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            data=gdf_year,
            get_fill_color="fill_color",
            pickable=True,
            auto_highlight=True,
            opacity=0.6,
        )
        layers.append(geojson_layer)
    elif heatmap_option == 'Project Heatmap':
        heatmap_layer = pdk.Layer(
            'HeatmapLayer',
            data=df_filtered,
            get_position='[longitude, latitude]',
            aggregation='MEAN',
            get_weight='capacity_mw',
        )
        layers.append(heatmap_layer)

    # Calculate radius based on settings
    if use_capacity_size:
        # Use logarithmic scaling for capacity
        df_filtered['radius'] = np.log1p(df_filtered['capacity_mw']) * 500 * size_factor
    else:
        df_filtered['radius'] = 1000 *size_factor

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_filtered,
        get_position='[longitude, latitude]',
        get_radius='radius',
        get_fill_color='[200, 30, 0, 160]',
        pickable=heatmap_option != 'State Energy Consumption',
        get_polygon="geometry"
    )
    layers.append(layer)
    tooltip = {
        "html": "<b>{name}</b><br/>Capacity: {capacity_mw} MW<br/>State: {state}<br/>Year: {year_commissioned}<br/>Commission Date: {commission_date}<br/>Type: {type}<br/>Status: {status}<br/>Developer: {developer}<br/>Owner: {owner}<br/>Supplier: {supplier}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    if heatmap_option == 'State Energy Consumption':
        tooltip = {
            "html": "<b>{NAME_1}</b><br/>Energy Consumption: {energy_requirement} GWh",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(**map_view_state),
        layers=layers,
        tooltip=tooltip,
    ))

    if heatmap_option == 'State Energy Consumption' and closest_year != selected_year:
        st.info(f"Estimated data (showing closest available year: {closest_year})")

def info_box_section(df_filtered):
    total_projects = len(df_filtered)
    total_capacity = df_filtered['capacity_mw'].sum()
    st.info(f"**Visible Projects:** {total_projects}  \n**Total Capacity:** {total_capacity:,.2f} MW")

# --- Table Section ---
def table_section(df_filtered):
    st.write('---')
    st.write('### Project Data Table')
    st.dataframe(df_filtered)

# --- Info Graphics Section (Bar Chart) ---
def info_graphics_section(df):
    # Filter out rows where year_commissioned is 0 or NaN
    df_filtered = df[df['year_commissioned'].notna() & (df['year_commissioned'] != 0)]
    
    # Projects per year chart
    bar_data = df_filtered.groupby('year_commissioned').size().reset_index(name='count')
    bar_chart = alt.Chart(bar_data).mark_bar().encode(
        x=alt.X('year_commissioned:O', title='Year Commissioned'),
        y=alt.Y('count:Q', title='Number of Projects'),
        tooltip=['year_commissioned', 'count']
    ).properties(width=600, height=200)
    st.write('---')
    st.write('### Projects Commissioned per Year')
    st.altair_chart(bar_chart, use_container_width=True)

    # Total Capacity per Year Chart
    cap_data = df_filtered.groupby('year_commissioned')['capacity_mw'].sum().reset_index()
    cap_chart = alt.Chart(cap_data).mark_line(point=True).encode(
        x=alt.X('year_commissioned:O', title='Year Commissioned'),
        y=alt.Y('capacity_mw:Q', title='Total Capacity (MW)'),
        tooltip=['year_commissioned', 'capacity_mw']
    ).properties(width=600, height=200)
    st.write('### Total Capacity Commissioned per Year')
    st.altair_chart(cap_chart, use_container_width=True)

    # Capacity Distribution Histogram
    st.write('### Project Capacity Distribution')
    hist_chart = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X('capacity_mw:Q', 
                title='Project Capacity (MW)',
                bin=alt.Bin(maxbins=50)),
        y=alt.Y('count()', title='Number of Projects'),
        tooltip=['count()', 'capacity_mw']
    ).properties(width=600, height=200)
    st.altair_chart(hist_chart, use_container_width=True)

# --- Main App ---
def main():
    # Load the data from Supabase
    df = load_data_from_supabase()
    year_min = int(df['year_commissioned'].min())
    year_max = int(df['year_commissioned'].max())
    min_capacity = int(df['capacity_mw'].min())
    max_capacity = int(df['capacity_mw'].max())

    if 'map_view' not in st.session_state:
        st.session_state['map_view'] = DEFAULT_MAP_VIEW.copy()

    header_section()
    selected_state, selected_type, selected_status, selected_developer, selected_owner, selected_supplier, capacity_range, reset, reset_map, heatmap_option, size_factor, use_capacity_size = sidebar(df, year_min, year_max, min_capacity, max_capacity)

    # Apply all filters except year
    df_filtered = df.copy()
    if selected_state:
        df_filtered = df_filtered[df_filtered['state'].isin(selected_state)]
    if selected_type:
        df_filtered = df_filtered[df_filtered['type'].isin(selected_type)]
    if selected_status:
        df_filtered = df_filtered[df_filtered['status'].isin(selected_status)]
    if selected_developer:
        df_filtered = df_filtered[df_filtered['developer'].isin(selected_developer)]
    if selected_owner:
        df_filtered = df_filtered[df_filtered['owner'].isin(selected_owner)]
    if selected_supplier:
        df_filtered = df_filtered[df_filtered['supplier'].isin(selected_supplier)]
    df_filtered = df_filtered[(df_filtered['capacity_mw'] >= capacity_range[0]) & (df_filtered['capacity_mw'] <= capacity_range[1])]

    # Year slider (right above the map)
    selected_year = year_slider_section(df, year_min, year_max)
    df_filtered = df_filtered[df_filtered['year_commissioned'] <= selected_year]

    # Map (with default view for India)
    if reset_map:
        st.session_state['map_view'] = DEFAULT_MAP_VIEW.copy()
    map_section(df_filtered, st.session_state['map_view'], heatmap_option, size_factor, use_capacity_size, selected_year)

    info_box_section(df_filtered)

    # Table section (now above the plot)
    table_section(df_filtered)

    # Info graphics section (bar chart)
    info_graphics_section(df)

def get_closest_year(selected_year, available_years):
    return min(available_years, key=lambda x: abs(x - selected_year))

if __name__ == '__main__':
    main()