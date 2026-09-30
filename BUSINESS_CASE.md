# Business Case

## Product concept

**AI4GOOD Urban Green Intelligence** converts repeated satellite or aerial observations into auditable indicators of urban vegetation coverage and change.

The commercial output is not “a JEPA model.” It is decision support for questions such as:

- Where is vegetation decreasing?
- Which zones should be inspected?
- Are greening interventions producing visible change?
- Which areas have persistently low canopy or vegetation coverage?
- Where do remotely sensed indicators disagree with expected conditions?

## Initial customer segments

### Municipalities
Use cases:
- monitor urban greening programs;
- prioritize inspections;
- compare districts;
- track change through time.

### ESG and environmental consultants
Use cases:
- evidence-backed vegetation indicators;
- project monitoring;
- recurring client reporting;
- screening before field audits.

### Real-estate and infrastructure
Use cases:
- monitor green-area commitments;
- detect landscaping loss;
- compare development phases;
- support environmental reporting.

### Industrial sites and logistics parks
Use cases:
- perimeter vegetation tracking;
- mitigation-area monitoring;
- recurring environmental audits.

## Value proposition

Traditional field surveys are valuable but costly to repeat at high frequency and broad geographic scale. Remote sensing can provide repeatable screening, while field validation remains the source of truth for ambiguous or consequential findings.

The proposed workflow therefore follows:

```text
remote sensing screening
        ↓
change / anomaly prioritization
        ↓
human review
        ↓
field inspection when needed
        ↓
verified record
```

## MVP outcome

The MVP should demonstrate that a user can:

1. select an urban zone;
2. view vegetation coverage for two dates;
3. identify estimated gain/loss;
4. see a transparent audit score;
5. understand why the zone was flagged;
6. export or record the finding for validation.

## Success metrics

Technical:
- IoU / F1 for vegetation segmentation;
- embedding-change separability;
- performance at 100%, 10%, 5%, and 1% label availability;
- calibration / uncertainty where applicable.

Business:
- analyst minutes per square kilometer;
- percentage of zones screened automatically;
- number of high-priority field visits generated;
- repeatability across dates;
- cost per monitored area.

## Commercial hypothesis

If self-supervised representations reduce labeling requirements while maintaining useful downstream accuracy, the platform can lower the cost of adapting vegetation-monitoring models to new cities and imagery sources.
