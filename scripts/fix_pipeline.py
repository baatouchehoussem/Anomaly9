path = 'C:/Users/baato/Downloads/My projects/Anomaly9/src/anomaly9/pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
old = 'import logging\nimport time as time_mod'
new = 'import logging\nfrom anomaly9.catalog import enrich_with_catalogs\nimport time as time_mod'
content = content.replace(old, new)
idx = content.find('__all__ = [')
if idx >= 0:
    bracket_count = 0
    end_idx = idx
    for i, ch in enumerate(content[idx:], idx):
        if ch == '[':
            bracket_count += 1
        elif ch == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i + 1
                break
    old_all = content[idx:end_idx]
    parts = []
    parts.append('__all__ = [')
    parts.append('    "LightCurveData",')
    parts.append('    "search_mast",')
    parts.append('    "ingest_mast_lightcurve",')
    parts.append('    "ingest_mast_lightcurve_multi_sector",')
    parts.append('    "clean_lightcurve",')
    parts.append('    "detrend_lightcurve",')
    parts.append('    "normalize_target_id",')
    parts.append('    "sanitize_filename",')
    parts.append('    "mask_momentum_dumps",')
    parts.append('    "stitch_lightcurves",')
    parts.append('    "enrich_with_catalogs",')
    parts.append(']')
    new_all = chr(10).join(parts)
    content = content[:idx] + new_all + content[end_idx:]
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('pipeline.py updated')
