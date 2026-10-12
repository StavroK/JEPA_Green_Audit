# M3d third-site candidate — Parque Niños Héroes

## Current evidence

The candidate AOI is centered near Parque Niños Héroes in northern Monterrey.

The local archive scan found:

- no overlap from the two downloaded RGB orthophoto ZIPs already used for Fundidora and La Pastora;
- two additional INEGI ZIPs that are elevation products rather than BIL RGB orthophotos.

External source checking places the AOI inside the Monterrey 1:50,000 sheet **G14C26**. Independent published work on nearby Ciudad Universitaria UANL explicitly reports use of the **INEGI 2007 digital orthophoto G14C26A1**, and another Monterrey study lists G14C26A1 among the 2007 1:10,000 orthophotos used for the metropolitan area.

Therefore the next acquisition target is:

> **INEGI 2007 orthophoto sheet G14C26A1**

This is an acquisition target, not yet a verified local source. The archive must still be downloaded from INEGI and checked against the AOI with `scripts/find_inegi_archive_for_aoi.py` before ingestion.

## Guardrail

Do not substitute the 2024 elevation products for the 2007 RGB orthophoto. M3d is a cross-site RGB transfer validation and should remain temporally and modality consistent with Fundidora/La Pastora.
