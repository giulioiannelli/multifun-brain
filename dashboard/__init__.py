"""multifun-brain dashboard (FastAPI backend + React frontend).

The backend reuses the ``multifunbrain`` library directly: it loads pipeline
result bundles, serialises them into JSON plot specs for interactive rendering,
and (later) drives ingestion/elaboration of new data folders. Nothing here
duplicates analysis logic — it is a thin presentation layer.
"""
