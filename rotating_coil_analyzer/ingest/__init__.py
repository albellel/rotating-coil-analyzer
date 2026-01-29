"""Ingest package - file readers and measurement discovery.

This package handles:
- Discovery of measurement folders and their structure
- Reading streaming binary files (*.bin)
- Reading plateau text files (*_raw_measurement_data.txt)
- Parsing Parameters.txt and FDIs table mapping

Key classes:
- MeasurementDiscovery: Scans folders and builds MeasurementCatalog
- StreamingReader: Reads continuous acquisition binary files
- PlateauReader: Reads DC plateau text files

Design principle:
- Readers produce validated SegmentFrame objects
- No synthetic time is created during ingestion
- All metadata is preserved and propagated
"""
