import os
import pandas as pd
from supabase import create_client, Client
from internal.config import settings

# Ensure you set these environment variables before running:
#   SUPABASE_URL: your Supabase project URL
#   SUPABASE_KEY: your Supabase anon or service role key
supabase_url = settings.SUPABASE_URL
supabase_key = settings.SUPABASE_KEY

if not supabase_url or not supabase_key:
    raise EnvironmentError('Please set SUPABASE_URL and SUPABASE_KEY environment variables.')

supabase: Client = create_client(supabase_url, supabase_key)

# Path to the CSV file
csv_path = 'start.csv'

def get_or_create_entity(name: str, role: str) -> int:
    """
    Fetches the entity_id for the given name and role, inserting a new record if necessary.
    """
    resp = (
        supabase.table('entities')
        .select('entity_id')
        .eq('name', name)
        .eq('role', role)
        .limit(1)
        .execute()
    )
    records = resp.data or []
    if records:
        return records[0]['entity_id']

    ins = supabase.table('entities').insert({'name': name, 'role': role}).execute()
    return ins.data[0]['entity_id']


def ingest_projects_from_csv(path: str):
    df = pd.read_csv(path)

    for idx, row in df.iterrows():
        # Build project record
        project = {
            'name': row['name'],
            'capacity_mw': row['capacity-(mw)'],
            'latitude': row['lat'],
            'longitude': row['lng'],
            'year_commissioned': int(row['start-year']) if pd.notna(row['start-year']) else None,
            'type': None,
        }

        # Insert project and retrieve project_id
        pr = supabase.table('projects').insert(project).execute()
        project_id = pr.data[0]['project_id']

        # Insert source record for project name
        source_url = str(row.get('url', '')).strip()
        if source_url and source_url.lower() != 'nan':
            src = supabase.table('sources').insert({'url': source_url}).execute()
            source_id = src.data[0]['source_id']
            supabase.table('source_fields').insert({
                'source_id': source_id,
                'project_id': project_id,
                'table_name': 'projects',
                'column_name': 'name'
            }).execute()

        # Link Owner if available
        owner_name = str(row.get('owner', '')).strip()
        if owner_name and owner_name.lower() != 'nan':
            owner_id = get_or_create_entity(owner_name, 'Owner')
            supabase.table('project_entities').insert({
                'project_id': project_id,
                'entity_id': owner_id
            }).execute()

        # Link Operator if available
        operator_name = str(row.get('operator', '')).strip()
        if operator_name and operator_name.lower() != 'nan':
            operator_id = get_or_create_entity(operator_name, 'Operator')
            supabase.table('project_entities').insert({
                'project_id': project_id,
                'entity_id': operator_id
            }).execute()

        print(f"Imported project '{project['name']}' (ID: {project_id})")


if __name__ == '__main__':
    ingest_projects_from_csv(csv_path)
