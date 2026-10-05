Busienss logic -
"Match fields using these rules: (1) exact name matches are high confidence, (2) same data type + similar sample values are medium confidence, (3) flag anything with no reasonable match as unmapped. For duplicates, treat records as the same customer if they share a phone number AND have similar names."

---------------------------------------------Command Prompt STEP 1------------------
1. gcloud auth application-default login
2. After login completes, run: gcloud auth application-default print-access-token If this prints a long token string (not an error), credentials are correctly saved on disk.
3. In that same Command Prompt (same window, same session), run: jupyter notebook This restarts Jupyter in a process that now has access to the credentials you just set up.
4. "gcloud auth application-default print-access-token shows — that'll confirm whether credentials are actually saved correctly before we go further.
5. "gcloud projects list" lists the projects
6. gcloud config set project {project id}
7.gcloud config get-value project - after config


------------------------------On the GCP service-------------
1. Create a new project
2. In the left sidebar or search bar at the top, search for 'Vertex AI' (it may also show as 'Agent Platform' in the newer UI). Click into it.
3. Inside Vertex AI, find 'Model Garden' in the left sidebar. Click it, then search for 'gemini flash' in the search bar. Make sure it is enabled
4. source_schema.csv, target_schema.csv, and sample_source_data.csv ,  migration_utils.py upload these 4 files in jupyter notebook for interaction and datamapping.




-----------------------------------------------Jupyter notebook commands Step 2-----------------
!pip install google-genai pandas

import migration_utils as mu

client = mu.get_client(project_id="project ID")

source_schema = mu.load_schema("source_schema.csv")
target_schema = mu.load_schema("target_schema.csv")
source_schema

mapping_df = mu.suggest_field_mappings(client, source_schema, target_schema)
mapping_df

sample_data = mu.load_schema("sample_source_data.csv")
quality_report = mu.flag_data_quality_issues(client, sample_data)
print(quality_report)

final_doc = mu.generate_mapping_document(client, mapping_df, quality_report)
mu.save_markdown(final_doc, "migration_mapping_report.md")
print(final_doc)


