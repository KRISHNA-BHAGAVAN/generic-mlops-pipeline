**What this file is for**
- It defines a strict “schema” for experiment configs (the YAML you load).
- It uses Pydantic v2 to validate that config values make sense.
- It also has a helper function that checks the config against the actual dataset (a pandas DataFrame).

**Key parts**
- Enums (`TaskType`, `SplitStrategy`) restrict certain fields to known values (e.g., only `"regression"` or `"classification"`).
- `ExperimentConfig` is the main model: if your YAML doesn’t match it, validation fails with clear errors.
- `validate_config_against_data()` checks the config vs. the dataset (columns exist, target type fits the task, enough rows).

---

**Validators (easy explanation)**

There are two kinds used here:

1) **Field validators**  
These run on a single field, but can *look at other fields* too.

- `validate_model_for_task` (on `model_type`)  
  It checks: is this model allowed for the chosen `task_type`?  
  Example: if task is `"regression"` but model is `"logistic_regression"`, it raises an error.

- `validate_metrics_for_task` (on `metrics`)  
  It checks: are all metrics valid for the task?  
  Example: `"accuracy"` is fine for classification, not regression.

How they work in plain words:
- Look up the task in `ALLOWED_MODELS` / `ALLOWED_METRICS`.
- If something doesn’t belong, raise a `ValueError` with a helpful message.

2) **Model validator**  
This runs *after* all fields are set, so it can check relationships between fields.

- `cross_field_validation`  
  It checks:
  - The `target_column` is not also listed in `feature_columns`.
  - If `split_strategy` is `"temporal"`, a `date_column` must be provided.

Think of this as “config-level logic” that depends on multiple fields.

---

**Why `model_config = {"use_enum_values": True}`?**  
This tells Pydantic to store enums as their string values (like `"regression"`) instead of enum objects. It makes the config simpler to use later.

---

**Dataset-level checks (not Pydantic)**
The `validate_config_against_data()` function checks the *actual data*:
- Target and feature columns exist in the DataFrame.
- Target dtype makes sense for the task.
- Dataset has at least 10 rows.

This is separate from the Pydantic model because it needs the real dataset.