# Assertions

The SDK provides assertion helpers for validating model predictions.

## Comprehensive Validation

### assert_valid_predictions

The main validation function that performs all checks:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04"] * 21,
    "location": ["Bokeo"] * 21,
    "samples": [[10.0, 12.0, 11.0]] * 21,
})

assert_valid_predictions(predictions, expected_rows=21)
```

This calls multiple underlying assertions to validate:

- Prediction shape
- Required columns (time_period, location)
- Samples column structure
- Consistent sample counts
- Numeric sample values

## Individual Assertions

### assert_prediction_shape

Verify predictions match the expected shape:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04", "2013-05"],
    "location": ["Bokeo", "Bokeo"],
    "samples": [[10.0, 12.0], [11.0, 13.0]],
})

future_data = DataFrame.from_dict({
    "time_period": ["2013-04", "2013-05"],
    "location": ["Bokeo", "Bokeo"],
})

assert_prediction_shape(predictions, future_data)
```

### assert_samples_column

Validate the samples column contains valid numeric lists:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04"],
    "location": ["Bokeo"],
    "samples": [[10.0, 12.0, 11.0]],
})

assert_samples_column(predictions, min_samples=1)
```

### assert_consistent_sample_counts

Ensure all rows have the same number of samples:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04", "2013-05"],
    "location": ["Bokeo", "Bokeo"],
    "samples": [[10.0, 12.0, 11.0], [15.0, 17.0, 16.0]],
})

assert_consistent_sample_counts(predictions)
```

### assert_numeric_samples

Type checking for sample values:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04"],
    "location": ["Bokeo"],
    "samples": [[10.0, 12.0, 11.0]],
})

assert_numeric_samples(predictions)
```

### assert_time_location_columns

Verify required columns are present:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04"],
    "location": ["Bokeo"],
    "samples": [[10.0, 12.0]],
})

assert_time_location_columns(predictions)
```

### assert_wide_format_predictions

Validate wide format predictions:

```python
wide_predictions = DataFrame.from_dict({
    "time_period": ["2013-04", "2013-05"],
    "location": ["Bokeo", "Bokeo"],
    "sample_0": [10.0, 15.0],
    "sample_1": [12.0, 17.0],
    "sample_2": [11.0, 16.0],
})

assert_wide_format_predictions(wide_predictions)
```

### assert_nonnegative_predictions

Verify all predictions are non-negative:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04"],
    "location": ["Bokeo"],
    "samples": [[10.0, 12.0, 11.0]],
})

assert_nonnegative_predictions(predictions)
```

### assert_no_nan_predictions

Check for missing values:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04"],
    "location": ["Bokeo"],
    "samples": [[10.0, 12.0, 11.0]],
})

assert_no_nan_predictions(predictions)
```

## Custom Exception

All assertion failures raise `PredictionValidationError`:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04"],
    "location": ["Bokeo"],
    # Missing samples - will fail validation
})

try:
    assert_valid_predictions(predictions)
except PredictionValidationError as e:
    print(f"Validation failed: {e}")
```

## Testing Patterns

### Basic Test

```python notest
import pytest
from chap_python_sdk.testing import (
    assert_valid_predictions,
    PredictionValidationError,
)


def test_predictions_are_valid():
    """Test that model produces valid predictions."""
    predictions = model.predict(data)

    # Will raise PredictionValidationError if invalid
    assert_valid_predictions(predictions)
```

### Testing for Failures

```python
import pytest

invalid_predictions = DataFrame.from_dict({
    "time_period": ["2013-04"],
    "location": ["Bokeo"],
    # Missing samples column
})

with pytest.raises(PredictionValidationError):
    assert_valid_predictions(invalid_predictions)
```

### Selective Validation

```python notest
def test_prediction_structure():
    """Test specific aspects of predictions."""
    predictions = model.predict(data)

    # Only check what matters for this test
    assert_time_location_columns(predictions)
    assert_samples_column(predictions, min_samples=10)
```

## When to Use Which Assertion

| Assertion | Use Case |
|-----------|----------|
| `assert_valid_predictions` | Comprehensive check in integration tests |
| `assert_prediction_shape` | Verify output matches expected dimensions |
| `assert_samples_column` | Validate sample structure and count |
| `assert_consistent_sample_counts` | Ensure uniform sample sizes |
| `assert_numeric_samples` | Type validation for samples |
| `assert_nonnegative_predictions` | Domain-specific constraint (e.g., counts) |
| `assert_no_nan_predictions` | Ensure complete predictions |

## Next Steps

- Learn about [Model Testing](model-testing.md)
- Check out the [API Reference](../api/assertions.md)
