Based on the test report for the FormFinder pipeline, there are two main issues causing test failures that need to be resolved. Below is a detailed explanation of the problems and the steps to fix them.

---

### Identified Issues

1. **Missing Dependency: PyYAML**
   - **Problem**: The dependencies test failed because the `pyyaml` package is not installed, as indicated by the error: `✗ pyyaml - NOT INSTALLED: No module named 'pyyaml'`.
   - **Impact**: This prevents the test from verifying all required packages, even though the pipeline might function without it in some cases if YAML parsing isn’t used outside the test.

2. **DataFetcher Initialization Error**
   - **Problem**: The `DataFetcher` test failed with an `AttributeError: 'dict' object has no attribute 'api'` at `DataFetcher.py`, line 294. This occurs because the test passes a dictionary (`test_config`) to `DataFetcher`, but the code tries to access `config.api` as an attribute instead of a dictionary key.
   - **Impact**: The `DataFetcher` class cannot initialize properly in the test environment due to this mismatch.

---

### Solutions

#### 1. Install Missing Dependency (PyYAML)
To resolve the missing `pyyaml` package:

- **Step 1**: Install `pyyaml` in your Python environment.
  Run the following command in your terminal:
  ```bash
  pip install pyyaml
  ```

- **Step 2**: Update `requirements.txt` to ensure this dependency is included for future installations.
  Run this command to append `pyyaml` to the file:
  ```bash
  echo pyyaml >> requirements.txt
  ```
  Alternatively, manually open `requirements.txt` in a text editor and add `pyyaml` on a new line if it’s not already present.

- **Verification**: After installation, rerun the test (`python .\test_full_pipeline.py`) to confirm the dependencies test now passes.

#### 2. Fix DataFetcher Configuration Access
The `DataFetcher` class expects a `DataFetcherConfig` object (likely a Pydantic model) but is being passed a dictionary in the test, and the code inside `DataFetcher` incorrectly uses attribute access instead of dictionary access.

- **Step 1**: Update the `DataFetcher` class to handle configuration correctly.
  Open `formfinder/DataFetcher.py` and locate line 294 (or the relevant section in `__init__`). You’ll likely see something like:
  ```python
  self.rate_limit_requests = config.api.rate_limit_requests
  ```
  Change it to use dictionary access, since `config.api` is a dictionary:
  ```python
  self.rate_limit_requests = config.api['rate_limit_requests']
  ```
  Review the rest of the `__init__` method and apply similar changes wherever `config.api` or `config.processing` is accessed as an attribute (e.g., `config.processing.league_ids` should become `config.processing['league_ids']`).

- **Step 2**: Ensure the test uses a `DataFetcherConfig` instance.
  Open `test_full_pipeline.py` and find the `test_data_fetcher` function (around line 32). The original code likely looks like:
  ```python
  test_config = {
      'api': {'rate_limit_requests': 10},
      'processing': {'league_ids': [1,2,3]}
  }
  fetcher = DataFetcher(test_config)
  ```
  Modify it to explicitly create a `DataFetcherConfig` instance:
  ```python
  from formfinder.config import DataFetcherConfig

  test_config = DataFetcherConfig(
      api={'rate_limit_requests': 10},
      processing={'league_ids': [1,2,3]}
  )
  fetcher = DataFetcher(test_config)
  ```
  This ensures the test passes a properly structured `DataFetcherConfig` object, consistent with how the main pipeline initializes `DataFetcher` (e.g., in `workflows.py`).

- **Why This Works**: 
  - In the main pipeline, `DataFetcherConfig` is created with dictionaries (via `model_dump()`), and its fields like `api` and `processing` are dictionaries. The test should mirror this structure.
  - Updating `DataFetcher` to use dictionary access aligns with the data type, preventing the `AttributeError`.

---

### Verification
After applying these fixes:
1. Run the test again:
   ```bash
   python .\test_full_pipeline.py
   ```
2. Check the output:
   - The **Dependencies** section should now show `✓ pyyaml` instead of `✗ pyyaml - NOT INSTALLED`.
   - The **DataFetcher** test should pass without the `AttributeError`.

---

### Additional Notes
- **Other Tests**: The remaining tests (File Structure, Configuration, DataProcessor, PredictorOutputter, and Notifier) passed, indicating those components are functioning correctly. The `0 matches` and `0 predictions` in `DataProcessor` and `PredictorOutputter` are likely due to test data and don’t indicate failures.
- **Consistency**: If you encounter similar attribute errors elsewhere, ensure all configuration accesses use dictionary syntax (e.g., `config['key']`) unless the config is explicitly a Pydantic model with attribute-style fields.

By installing `pyyaml` and correcting the configuration handling in `DataFetcher`, the FormFinder pipeline test should pass all 7 checks successfully.