# label_cleaner

Modular rewrite of the original experiment codebase.

## Architecture

### Foldered Modules (primary)

- `core/`
  - `models.py` — typed dataclasses (`PreparedSplit`, `NoiseBundle`, `MethodCurves`)
  - `prep.py` — dataset split preparation (`prepare_fixed_split`)
- `data/`
  - `datasets.py` — dataset loading + metadata (`DatasetInfo`, `load_dataset`)
- `methods/`
  - `noise.py` — noise injectors (`inject_outlier`, `inject_rnd_label`, `inject_nnar`, `inject_mnar`)
  - `pipelines.py` — pipeline factories (`make_pipeline_a`, `make_pipeline_b`)
  - `cleaning.py` — methodology functions (`clean_datascope`, `clean_random`, `clean_cleanlab`)
- `orchestration/`
  - `experiments.py` — run functions per noise type
  - `catalog.py` — experiment/pipeline registry
- `services/`
  - `service.py` — microservice-style facade (`LabelCleanerService`)

### Compatibility Wrappers

Top-level modules (`datasets.py`, `noise.py`, `pipelines.py`, `cleaning.py`, etc.)
are retained as import wrappers for backward compatibility.

## Example

```python
from pathlib import Path
from label_cleaner.service import LabelCleanerService

svc = LabelCleanerService(datasets_dir=Path("datasets"))
result = svc.run(
    dataset="adult",
    noise_type="rnd_label",
    pipeline_key="p1a",
    noise_level=0.2,
)

print(result["baseline"])          # noisy baseline accuracy
print(result["datascope"][-1])     # accuracy after 100% DataScope cleaning
print(result["cleanlab"][-1])      # accuracy after 100% CleanLab cleaning
```

## Notes

- All methodology functions are isolated and reusable.
- Service API is ready to expose via REST/gRPC later if needed.
- Current implementation preserves behavior from the original scripts.
