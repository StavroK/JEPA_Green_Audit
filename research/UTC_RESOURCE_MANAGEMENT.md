# UTC management zones and resource prioritization

## Objective

Urban Tree Canopy should not be only a citywide percentage. Users need to know
**where canopy is low, where it is changing, and which management areas require
attention**.

The product should therefore calculate UTC at multiple spatial levels:

- municipality;
- neighborhood / colonia;
- park;
- campus;
- industrial or logistics site;
- census/statistical area;
- custom user-drawn management zone.

## Core area-level metrics

For every reporting zone:

- UTC %;
- canopy area (ha);
- valid analysis area (ha);
- UTC change (percentage points) when two comparable dates exist;
- canopy gain (ha);
- canopy loss (ha);
- optional deficit to a **user-defined** UTC target.

A target must not be hard-coded globally. Appropriate canopy targets differ by
urban form, climate, land use, infrastructure, ownership, and policy.

## Resource-management outputs

The dashboard should make it possible to answer:

- Which zones have the lowest UTC?
- Which zones are losing canopy fastest?
- Where is canopy stable or increasing?
- Which zones have the largest canopy deficit relative to a selected target?
- Which areas should be inspected first?
- Where should field teams verify suspected loss?
- Where are investments having measurable effect?

## Priority is not planting suitability

Low UTC or canopy loss should create a **management attention signal**, not an
automatic recommendation to plant trees.

Planting feasibility requires additional layers such as:

- land ownership / right-of-way;
- impervious surfaces and buildings;
- roads and mobility infrastructure;
- utilities;
- soil / water availability;
- safety and visibility constraints;
- land use;
- local climate/species constraints;
- field validation.

The system should keep these separate:

```text
Canopy condition / deficit
          +
Change / loss signal
          +
Feasibility constraints
          +
User priorities / budget
          ↓
Management priority
```

## Implementation

`scripts/aggregate_utc_by_zones.py` aggregates a validated binary canopy mask
over a user-supplied GeoJSON FeatureCollection.

It writes:

- GeoJSON for mapping;
- CSV for dashboard/BI use;
- optional target-deficit metrics.

This is the bridge between the remote-sensing pipeline and resource allocation.
