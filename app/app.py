import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt

# --- Constants ---
DEFAULT_MAP_VIEW = {
    'latitude': 22.5937,  # Center of India
    'longitude': 78.9629,
    'zoom': 4,
    'pitch': 0
}

# --- Title and Subtitle ---
def header_section():
    st.title("🌞 Solar Detective: India's Solar Project Dashboard")
    st.write("Explore, filter, and visualize India's solar infrastructure projects with interactive maps and analytics.")

# --- Sidebar Filters ---
def sidebar_filters(df, year_min, year_max, min_capacity, max_capacity):
    st.sidebar.header('Filter Projects')
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        reset = st.button('Reset Filters')
    with col_b:
        reset_map = st.button('Reset Map')
    # State filter
    states = ['All'] + sorted(df['state'].dropna().unique().tolist())
    if 'state' not in st.session_state or reset:
        st.session_state['state'] = 'All'
    selected_state = st.sidebar.selectbox('State', states, key='state')
    # Project type filter
    types = ['All'] + sorted(df['type'].dropna().unique().tolist())
    if 'type' not in st.session_state or reset:
        st.session_state['type'] = 'All'
    selected_type = st.sidebar.selectbox('Project Type', types, key='type')
    # Status filter
    statuses = ['All'] + sorted(df['status'].dropna().unique().tolist())
    if 'status' not in st.session_state or reset:
        st.session_state['status'] = 'All'
    selected_status = st.sidebar.selectbox('Status', statuses, key='status')
    # Capacity range filter
    if 'capacity' not in st.session_state or reset:
        st.session_state['capacity'] = (min_capacity, max_capacity)
    capacity_range = st.sidebar.slider('Capacity (MW)', min_capacity, max_capacity, key='capacity')
    # Year filter (for resetting)
    if 'year_slider' not in st.session_state or reset:
        st.session_state['year_slider'] = year_max
    return selected_state, selected_type, selected_status, capacity_range, reset, reset_map

# --- Year Slider ---
def year_slider_section(df, year_min, year_max):
    st.write('### Filter by Commissioned Year')
    selected_year = st.slider('Show projects commissioned up to year:', year_min, year_max, key='year_slider')
    return selected_year

# --- Map Section ---
def map_section(df_filtered, map_view_state):
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_filtered,
        get_position='[longitude, latitude]',
        get_radius=50000,
        get_fill_color='[200, 30, 0, 160]',
        pickable=True
    )
    tooltip = {
        "html": "<b>{name}</b><br/>Capacity: {capacity_mw} MW<br/>State: {state}<br/>Year: {year_commissioned}<br/>Commission Date: {commission_date}<br/>Type: {type}<br/>Developer: {developer}<br/>Status: {status}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(**map_view_state),
        layers=[layer],
        tooltip=tooltip
    ))

# --- Table Section ---
def table_section(df_filtered):
    st.write('---')
    st.write('### Project Data Table')
    st.dataframe(df_filtered)

# --- Info Graphics Section (Bar Chart) ---
def info_graphics_section(df):
    bar_data = df.groupby('year_commissioned').size().reset_index(name='count')
    bar_chart = alt.Chart(bar_data).mark_bar().encode(
        x=alt.X('year_commissioned:O', title='Year Commissioned'),
        y=alt.Y('count:Q', title='Number of Projects'),
        tooltip=['year_commissioned', 'count']
    ).properties(width=600, height=200)
    st.write('---')
    st.write('### Projects Commissioned per Year')
    st.altair_chart(bar_chart, use_container_width=True)

# --- Main App ---
def main():
    # Load the sample data
    csv_path = 'data/sample_projects.csv'
    df = pd.read_csv(csv_path)
    year_min = int(df['year_commissioned'].min())
    year_max = int(df['year_commissioned'].max())
    min_capacity = int(df['capacity_mw'].min())
    max_capacity = int(df['capacity_mw'].max())

    if 'map_view' not in st.session_state:
        st.session_state['map_view'] = DEFAULT_MAP_VIEW.copy()

    header_section()
    selected_state, selected_type, selected_status, capacity_range, reset, reset_map = sidebar_filters(df, year_min, year_max, min_capacity, max_capacity)

    if reset_map:
        st.session_state['map_view'] = DEFAULT_MAP_VIEW.copy()

    # Apply all filters except year
    df_filtered = df.copy()
    if selected_state != 'All':
        df_filtered = df_filtered[df_filtered['state'] == selected_state]
    if selected_type != 'All':
        df_filtered = df_filtered[df_filtered['type'] == selected_type]
    if selected_status != 'All':
        df_filtered = df_filtered[df_filtered['status'] == selected_status]
    df_filtered = df_filtered[(df_filtered['capacity_mw'] >= capacity_range[0]) & (df_filtered['capacity_mw'] <= capacity_range[1])]

    # Year slider (right above the map)
    selected_year = year_slider_section(df, year_min, year_max)
    df_filtered = df_filtered[df_filtered['year_commissioned'] <= selected_year]

    # Map (with default view for India)
    map_section(df_filtered, st.session_state['map_view'])

    # Table section (now above the plot)
    table_section(df_filtered)

    # Info graphics section (bar chart)
    info_graphics_section(df)

if __name__ == '__main__':
    main()