path = 'C:/Users/baato/Downloads/My projects/Anomaly9/src/anomaly9/pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_import = "from __future__ import annotations

import logging"
new_import = "from __future__ import annotations

import logging
from anomaly9.catalog import enrich_with_catalogs"
content = content.replace(old_import, new_import)

old_all = "__all__ = ["
    "LightCurveData",
    "search_mast",
    "ingest_mast_lightcurve",
    "ingest_mast_lightcurve_multi_sector",
    "clean_lightcurve",
    "detrend_lightcurve",
    "normalize_target_id",
    "sanitize_filename",
    "mask_momentum_dumps",
    "stitch_lightcurves",
]"
new_all = "__all__ = ["
    "LightCurveData",
    "search_mast",
    "ingest_mast_lightcurve",
    "ingest_mast_lightcurve_multi_sector",
    "clean_lightcurve",
    "detrend_lightcurve",
    "normalize_target_id",
    "sanitize_filename",
    "mask_momentum_dumps",
    "stitch_lightcurves",
    "enrich_with_catalogs",
]"
content = content.replace(old_all, new_all)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("pipeline.py updated")
