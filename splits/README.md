# Split Manifests

Committed dataset splits use versioned JSON manifests under a dataset-specific directory. No real sequence identifiers are defined yet; manifests will be added only after data access and protocol approval.

Schema version 1 has this shape:

```json
{
  "schema_version": 1,
  "dataset": "scared",
  "dataset_version": null,
  "split_name": "scared_v1",
  "train": [],
  "validation": [],
  "test": [],
  "notes": ""
}
```

Identifiers must be unique within each list and must not overlap across train, validation, and test. Hashing sorts IDs within each split, so ordering is not scientifically meaningful. The hash excludes the file path, split name, notes, and timestamps.

