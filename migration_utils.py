"""
migration_utils.py
Helper functions for the AI-Assisted Data Migration Mapping Tool (using Gemini via Vertex AI).

Workflow:
  1. Load source and target schema CSVs
  2. Ask Gemini to suggest field mappings (structured JSON)
  3. Load sample source data and ask Claude to flag data quality issues
  4. Generate a final migration mapping document
"""

import json
import re
import pandas as pd
from google import genai

# Gemini models on Vertex AI are covered by your $300 Google Cloud free trial credit
# (unlike partner "model as a service" models such as Claude, which are billed separately).
MODEL = "gemini-2.5-flash"  # fast + cheap, good for this kind of structured task
REGION = "us-central1"  # change if your project uses a different default region


def get_client(project_id, region=REGION):
    """
    Create a Gemini client via Vertex AI, authenticated using your Google Cloud credentials
    (no API key needed, and usage draws from your $300 free trial credit).

    Before running this, authenticate locally with:
        gcloud auth application-default login
    (requires the Google Cloud CLI: https://cloud.google.com/sdk/docs/install)
    """
    return genai.Client(vertexai=True, project=project_id, location=region)


def load_schema(csv_path):
    """Load a schema CSV (field_name, data_type, sample_value, description)."""
    return pd.read_csv(csv_path)


def _extract_json(text):
    """Pull a JSON array/object out of an LLM text response, tolerating ```json fences."""
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("Could not find JSON in response:\n" + text)


def suggest_field_mappings(client, source_df, target_df, model=MODEL):
    """
    Ask Gemini to map each source field to the best-matching target field.
    Returns a pandas DataFrame: source_field, target_field, confidence, transformation_needed, notes
    """
    source_json = source_df.to_json(orient="records")
    target_json = target_df.to_json(orient="records")

    prompt = f"""You are a data migration expert. Map each SOURCE field to the single best-matching
TARGET field based on field name, data type, sample value, and description.

SOURCE SCHEMA:
{source_json}

TARGET SCHEMA:
{target_json}

For each source field, return an object with:
- source_field: the source field name
- target_field: best matching target field name (or null if no good match exists)
- confidence: "high", "medium", or "low"
- transformation_needed: a short description of any transformation required (e.g. "split full name into FirstName/LastName", "convert state code to full state name", "none")
- notes: any other useful note for a migration analyst

Respond with ONLY a JSON array of these objects, no other text, no markdown fences.
"""

    response = client.models.generate_content(model=model, contents=prompt)
    text = response.text
    data = _extract_json(text)
    return pd.DataFrame(data)


def flag_data_quality_issues(client, sample_data_df, model=MODEL):
    """
    Ask Gemini to review sample source data and flag likely data quality issues:
    duplicates, missing values, inconsistent formats.
    Returns the raw text analysis (printable / saveable as-is).
    """
    sample_json = sample_data_df.to_json(orient="records")

    prompt = f"""You are a data quality analyst reviewing a sample of customer records before migration.

SAMPLE DATA:
{sample_json}

Identify:
1. Likely duplicate customer records (same person, different CUST_ID) - explain why you think so
2. Missing or incomplete fields that would block clean migration
3. Inconsistent formats (e.g. inconsistent phone formats, malformed emails)
4. Any other data quality risks

Present this as a clear, structured summary a business analyst could paste into a migration report.
"""

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def generate_mapping_document(client, mapping_df, quality_report, model=MODEL):
    """
    Ask Gemini to generate a clean, professional migration mapping document
    combining the field mapping table and the data quality findings.
    Returns markdown text.
    """
    mapping_json = mapping_df.to_json(orient="records")

    prompt = f"""Create a professional Data Migration Mapping Document in markdown format using the
information below. Include these sections: Overview, Field Mapping Table, Data Quality Findings,
Recommended Next Steps.

FIELD MAPPINGS:
{mapping_json}

DATA QUALITY FINDINGS:
{quality_report}

Write it as a business analyst would present it to a project team - clear, structured, and ready to share.
"""

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def save_markdown(text, path):
    """Save text to a markdown file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {path}")

