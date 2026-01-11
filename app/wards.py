from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import shapefile  # type: ignore
except Exception:  # pragma: no cover
    shapefile = None  # type: ignore

SHAPEFILE_PATH = Path(__file__).with_name("kenya_wards.shp")

FALLBACK_WARDS: List[Dict[str, float | str]] = [
    {"id": "wote", "name": "Wote Ward", "subcounty": "Makueni", "latitude": -1.7803, "longitude": 37.6283},
    {"id": "mavindini", "name": "Mavindini Ward", "subcounty": "Makueni", "latitude": -1.9435, "longitude": 37.7418},
    {"id": "kitise_kithuki", "name": "Kitise/Kithuki Ward", "subcounty": "Makueni", "latitude": -2.0554, "longitude": 37.7138},
    {"id": "kathonzweni", "name": "Kathonzweni Ward", "subcounty": "Makueni", "latitude": -1.8974, "longitude": 37.6181},
    {"id": "ukia", "name": "Ukia Ward", "subcounty": "Kaiti", "latitude": -1.4602, "longitude": 37.3301},
    {"id": "kee", "name": "Kee Ward", "subcounty": "Kaiti", "latitude": -1.5368, "longitude": 37.3965},
    {"id": "kilungu", "name": "Kilungu Ward", "subcounty": "Kaiti", "latitude": -1.5557, "longitude": 37.2768},
    {"id": "ilima", "name": "Ilima Ward", "subcounty": "Kaiti", "latitude": -1.4425, "longitude": 37.4321},
    {"id": "tulimani", "name": "Tulimani Ward", "subcounty": "Mbooni", "latitude": -1.6141, "longitude": 37.3689},
    {"id": "kalawa", "name": "Kalawa Ward", "subcounty": "Mbooni", "latitude": -1.7712, "longitude": 37.4175},
    {"id": "kasikeu", "name": "Kasikeu Ward", "subcounty": "Kilome", "latitude": -1.7468, "longitude": 37.1828},
    {"id": "kilome", "name": "Kilome Ward", "subcounty": "Kilome", "latitude": -1.8227, "longitude": 37.2567},
    {"id": "kiima_kiu", "name": "Kiima Kiu/Kalanzoni Ward", "subcounty": "Kilome", "latitude": -1.9707, "longitude": 37.3038},
    {"id": "makindu", "name": "Makindu Ward", "subcounty": "Kibwezi West", "latitude": -2.2834, "longitude": 37.8281},
    {"id": "nguumo", "name": "Nguumo Ward", "subcounty": "Kibwezi West", "latitude": -2.2628, "longitude": 37.7426},
    {"id": "emali_mulala", "name": "Emali/Mulala Ward", "subcounty": "Kibwezi West", "latitude": -2.0578, "longitude": 37.4714},
    {"id": "masongaleni", "name": "Masongaleni Ward", "subcounty": "Kibwezi West", "latitude": -2.4025, "longitude": 37.7016},
    {"id": "kikumbulyu_north", "name": "Kikumbulyu North Ward", "subcounty": "Kibwezi West", "latitude": -2.4986, "longitude": 37.7142},
    {"id": "kikumbulyu_south", "name": "Kikumbulyu South Ward", "subcounty": "Kibwezi West", "latitude": -2.5987, "longitude": 37.7431},
    {"id": "mtito_andei", "name": "Mtito Andei Ward", "subcounty": "Kibwezi East", "latitude": -2.6889, "longitude": 38.1673},
    {"id": "thange", "name": "Thange Ward", "subcounty": "Kibwezi East", "latitude": -2.3336, "longitude": 37.9027},
    {"id": "ivingoni_nzambani", "name": "Ivingoni/Nzambani Ward", "subcounty": "Kibwezi East", "latitude": -2.5085, "longitude": 37.9938},
    {"id": "masongaleni_east", "name": "Masongaleni East Ward", "subcounty": "Kibwezi East", "latitude": -2.3652, "longitude": 38.024},
]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "ward"


def _find_field(field_names: List[str], keywords: List[str]) -> Optional[str]:
    for keyword in keywords:
        for name in field_names:
            if keyword in name.lower():
                return name
    return None


def _load_from_shapefile() -> List[Dict[str, float | str]]:
    if shapefile is None:
        print("[wards] pyshp not installed; falling back to static registry.")
        return []
    if not SHAPEFILE_PATH.exists():
        print(f"[wards] Shapefile not found at {SHAPEFILE_PATH}; using static registry.")
        return []
    try:
        reader = shapefile.Reader(str(SHAPEFILE_PATH))
    except shapefile.ShapefileException as exc:  # pragma: no cover
        print(f"[wards] Failed to parse shapefile: {exc}")
        return []

    fields = [field[0] for field in reader.fields[1:]]  # drop deletion flag
    ward_field = _find_field(fields, ["ward"])
    county_field = _find_field(fields, ["county"])
    subcounty_field = _find_field(fields, ["subcounty", "constituency", "sub_county"])

    if not ward_field or not county_field:
        print("[wards] Shapefile missing ward/county attribute fields; using static registry.")
        return []

    wards: List[Dict[str, float | str]] = []
    seen_ids: Dict[str, int] = {}
    for shape_record in reader.iterShapeRecords():
        record = shape_record.record
        if hasattr(record, "as_dict"):
            attrs = record.as_dict()  # type: ignore[attr-defined]
        else:
            attrs = dict(zip(fields, record))

        county_name = str(attrs.get(county_field, "")).strip()
        if "makueni" not in county_name.lower():
            continue

        ward_name = str(attrs.get(ward_field, "")).strip()
        if not ward_name:
            continue

        ward_id = _slugify(ward_name)
        if ward_id in seen_ids:
            seen_ids[ward_id] += 1
            ward_id = f"{ward_id}_{seen_ids[ward_id]}"
        else:
            seen_ids[ward_id] = 1

        subcounty_name = str(attrs.get(subcounty_field, "")).strip() if subcounty_field else ""
        bbox = shape_record.shape.bbox
        if not bbox or len(bbox) != 4:
            continue
        minx, miny, maxx, maxy = bbox
        longitude = float((minx + maxx) / 2.0)
        latitude = float((miny + maxy) / 2.0)

        wards.append(
            {
                "id": ward_id,
                "name": ward_name,
                "subcounty": subcounty_name or "Makueni",
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
            }
        )

    if not wards:
        print("[wards] No wards derived from shapefile; using static registry.")
        return []

    wards.sort(key=lambda entry: str(entry["name"]))
    return wards


WARD_REGISTRY: List[Dict[str, float | str]] = _load_from_shapefile() or FALLBACK_WARDS
WARD_LOOKUP: Dict[str, Dict[str, float | str]] = {ward["id"]: ward for ward in WARD_REGISTRY}
