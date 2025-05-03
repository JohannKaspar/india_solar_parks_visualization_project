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
    'zoom': 3.3,
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

    # Join with entities to get operator, owner, and supplier information
    entities_response = supabase.table('entities').select('*').execute()
    entities_data = entities_response.data
    entities_df = pd.DataFrame(entities_data)
    project_entities_response = supabase.table('project_entities').select('*').execute()
    project_entities_data = project_entities_response.data
    project_entities_df = pd.DataFrame(project_entities_data)
    # Merge entities with project_entities
    project_entities_df = project_entities_df.merge(entities_df, left_on='entity_id', right_on='entity_id', how='left')
    # Group by project_id and role to get lists of operators, owners, and suppliers
    operators = project_entities_df[project_entities_df['role'] == 'Operator'].groupby('project_id')['name'].apply(list).reset_index(name='operator')
    owners = project_entities_df[project_entities_df['role'] == 'Owner'].groupby('project_id')['name'].apply(list).reset_index(name='owner')
    suppliers = project_entities_df[project_entities_df['role'] == 'Supplier'].groupby('project_id')['name'].apply(list).reset_index(name='supplier')

    # Helper function to clean up stringified lists
    def clean_list(x):
        if not x or not isinstance(x, list):
            return np.nan
        if len(x) == 1 and isinstance(x[0], str):
            try:
                # Try to evaluate the string as a list
                evaluated = eval(x[0])
                if isinstance(evaluated, list):
                    if len(evaluated) == 0:
                        return np.nan
                    return evaluated
            except:
                # If evaluation fails, split by common delimiters
                if ';' in x[0]:
                    return [item.strip() for item in x[0].split(';')]
                elif ',' in x[0]:
                    return [item.strip() for item in x[0].split(',')]
        if len(x) == 1 and isinstance(x[0], list):
            if len(x[0]) == 0:
                return np.nan
            else:
                return x[0]
        if isinstance(x, str) and len(x) == 0:
            return np.nan
        if isinstance(x, str):
            return [x]
        return x

    # Clean up the lists
    operators['operator'] = operators['operator'].apply(clean_list)
    owners['owner'] = owners['owner'].apply(clean_list)
    suppliers['supplier'] = suppliers['supplier'].apply(clean_list)

    # Convert lists to strings for display
    operators['operator_str'] = operators['operator'].apply(lambda x: ', '.join(x) if x and isinstance(x, list) else 'N/A')
    owners['owner_str'] = owners['owner'].apply(lambda x: ', '.join(x) if x and isinstance(x, list) else 'N/A')
    suppliers['supplier_str'] = suppliers['supplier'].apply(lambda x: ', '.join(x) if x and isinstance(x, list) else 'N/A')

    # Merge with main df
    df = df.merge(operators, on='project_id', how='left')
    df = df.merge(owners, on='project_id', how='left')
    df = df.merge(suppliers, on='project_id', how='left')
    
    # Fill NaN with N/A
    df['operator_str'] = df['operator_str'].fillna('N/A')
    df['operator'] = df['operator']
    df['owner_str'] = df['owner_str'].fillna('N/A')
    df['owner'] = df['owner']
    df['supplier_str'] = df['supplier_str'].fillna('N/A')
    df['supplier'] = df['supplier']

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

    # Operator filter
    all_operators = []
    for operator_list in df['operator']:
        if operator_list and isinstance(operator_list, list):
            all_operators.extend(operator_list)
        elif np.isnan(operator_list):
            all_operators.append('N/A')
    operators = sorted(set(all_operators))
    if 'operator' not in st.session_state or reset:
        st.session_state['operator'] = []
    selected_operator = st.sidebar.multiselect('Operator', operators, key='operator')

    # Owner filter
    all_owners = []
    for owner_list in df['owner']:
        if isinstance(owner_list, list) and len(owner_list) > 0:
            all_owners.extend(owner_list)
        elif isinstance(owner_list, list) and len(owner_list) == 0 or np.isnan(owner_list):
            all_owners.append('N/A')
    owners = sorted(set(all_owners))
    if 'owner' not in st.session_state or reset:
        st.session_state['owner'] = []
    selected_owner = st.sidebar.multiselect('Owner', owners, key='owner')

    # Supplier filter
    all_suppliers = []
    for supplier_list in df['supplier']:
        if supplier_list and isinstance(supplier_list, list):
            all_suppliers.extend(supplier_list)
        elif np.isnan(supplier_list):
            all_suppliers.append('N/A')
    suppliers = sorted(set(all_suppliers))
    if 'supplier' not in st.session_state or reset:
        st.session_state['supplier'] = []
    selected_supplier = st.sidebar.multiselect('Supplier', suppliers, key='supplier')
    # Capacity range filter
    if 'capacity' not in st.session_state or reset:
        st.session_state['capacity'] = (min_capacity, max_capacity)
    capacity_range = st.sidebar.slider('Capacity (MW)', min_capacity, max_capacity, key='capacity')
    # Year filter (for resetting)
    if 'year_slider' not in st.session_state or reset:
        st.session_state['year_slider'] = (year_min, year_max)

    st.sidebar.markdown("---")  # Add a divider line
    st.sidebar.markdown("### Map Settings")  # Add a header for map settings

    # Dot size settings
    size_factor = st.sidebar.slider('Marker Size', min_value=1, max_value=20, value=10, step=1)
    use_capacity_size = st.sidebar.checkbox('Scale marker size by project capacity', value=False)
    
    # Heatmap selection
    heatmap_option = st.sidebar.selectbox('Heatmap', ['No Heatmap', 'Project Heatmap', 'Solar Resource (GHI)', 'State Energy Consumption'], key='heatmap_option')
    return selected_state, selected_type, selected_status, selected_operator, selected_owner, selected_supplier, capacity_range, reset, reset_map, heatmap_option, size_factor, use_capacity_size

# --- Year Slider ---
def year_slider_section(df, year_min, year_max):
    # Filter out rows where year_commissioned is 0 or NaN
    df_filtered = df[df['year_commissioned'].notna() & (df['year_commissioned'] > 1900)]
    year_min = int(df_filtered['year_commissioned'].min())
    year_max = int(df_filtered['year_commissioned'].max())
    st.write('### Filter by Commissioned Year')
    selected_year_range = st.slider(
        'Show projects commissioned between years:',
        year_min, year_max, key='year_slider')
    return selected_year_range

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
        df_filtered['radius'] = 1000 * size_factor

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
        "html": "<b>{name}</b><br/>Capacity: {capacity_mw} MW<br/>State: {state}<br/>Year: {year_commissioned}<br/>Commission Date: {commission_date}<br/>Type: {type}<br/>Status: {status}<br/>Operator: {operator}<br/>Owner: {owner_str}<br/>Supplier: {supplier}",
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
    # <iframe src="https://www.google.com/maps/d/u/0/embed?mid=1ssxg5qgL8pDlVKV4Jcc4axK_O9s2DYm2&ehbc=2E312F" width="100%" height="480"></iframe>
    # st.components.v1.iframe("https://www.google.com/maps/d/u/0/embed?mid=1ssxg5qgL8pDlVKV4Jcc4axK_O9s2DYm2&ehbc=2E312F")

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
    st.dataframe(df_filtered[['name', 'state', 'year_commissioned', 'commission_date', 'type', 'status', 'operator', 'owner', 'supplier', 'capacity_mw']])

# --- Info Graphics Section (Bar Chart) ---
def info_graphics_section(df):
    # Filter out rows where year_commissioned is 0 or NaN
    df_filtered = df[df['year_commissioned'].notna() & (df['year_commissioned'] > 1900)]
    
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
    year_min = int((df[df['year_commissioned'] > 1900]['year_commissioned']).min())
    year_max = int((df[df['year_commissioned'] > 1900]['year_commissioned']).max())
    min_capacity = int(df['capacity_mw'].min())
    max_capacity = int(df['capacity_mw'].max())

    if 'map_view' not in st.session_state:
        st.session_state['map_view'] = DEFAULT_MAP_VIEW.copy()

    header_section()
    selected_state, selected_type, selected_status, selected_operator, selected_owner, selected_supplier, capacity_range, reset, reset_map, heatmap_option, size_factor, use_capacity_size = sidebar(df, year_min, year_max, min_capacity, max_capacity)

    # Apply all filters except year
    df_filtered = df.copy()
    if selected_state:
        df_filtered = df_filtered[df_filtered['state'].isin(selected_state)]
    if selected_type:
        df_filtered = df_filtered[df_filtered['type'].isin(selected_type)]
    if selected_status:
        df_filtered = df_filtered[df_filtered['status'].isin(selected_status)]
    if selected_supplier:
        df_filtered = df_filtered[df_filtered['supplier'].apply(lambda x: match_selected(x, selected_supplier))]
    if selected_operator:
        df_filtered = df_filtered[df_filtered['operator'].apply(lambda x: match_selected(x, selected_operator))]
    if selected_owner:
        df_filtered = df_filtered[df_filtered['owner'].apply(lambda x: match_selected(x, selected_owner))]
    df_filtered = df_filtered[(df_filtered['capacity_mw'] >= capacity_range[0]) & (df_filtered['capacity_mw'] <= capacity_range[1])]

    # Year slider (right above the map)
    selected_year_range = year_slider_section(df, year_min, year_max)
    df_filtered = df_filtered[
        (df_filtered['year_commissioned'] >= selected_year_range[0]) &
        (df_filtered['year_commissioned'] <= selected_year_range[1])
    ]

    # Map (with default view for India)
    if reset_map:
        st.session_state['map_view'] = DEFAULT_MAP_VIEW.copy()
    map_section(df_filtered, st.session_state['map_view'], heatmap_option, size_factor, use_capacity_size, selected_year_range[1])

    info_box_section(df_filtered)

    # Table section (now above the plot)
    table_section(df_filtered)

    # Info graphics section (bar chart)
    info_graphics_section(df)

def get_closest_year(selected_year, available_years):
    return min(available_years, key=lambda x: abs(x - selected_year))

def match_selected(x, selected):
    if isinstance(x, list):
        return any(val in x for val in selected)
    elif pd.isna(x):
        return any(val == 'N/A' for val in selected)
    return False

if __name__ == '__main__':
    main()