# Documentation Illustration Scripts

This directory contains Python scripts for generating statistical illustrations used in the documentation.

## Overview

The scripts use **Altair** (declarative statistical visualization library) to create high-quality plots and diagrams
that are embedded in the markdown documentation files.

## Scripts

### `generate_bootstrap_illustrations.py`

Generates illustrations for `docs/bootstrapping-for-time-series.md`:

- Classical bootstrap process visualization
- Comparison of i.i.d. vs time series data
- Temporal structure destruction in classical bootstrap
- Moving block bootstrap illustration
- Block length trade-off visualization
- Seasonal block bootstrap
- ACF plot for block length selection
- Bootstrap confidence interval distribution

### `generate_residual_bootstrap_illustrations.py`

Generates illustrations for `docs/residual-bootstrapping-time-series.md`:

- Original data with fitted model
- Residuals plot
- Bootstrap sample comparison
- Residual ACF plot
- Block bootstrap illustration
- Residual diagnostic plots (homoscedasticity check)
- QQ plot for normality
- Residuals vs fitted values
- Bootstrap distribution comparison
- Forecast intervals

### `generate_quantile_regression_illustrations.py`

Generates illustrations for `docs/quantile-regression.md`:

- OLS vs quantile regression comparison
- Check loss function (pinball loss)
- Heteroscedastic data comparison
- Quantile prediction intervals
- Time series quantile regression
- Bootstrap distribution of coefficients
- Crossing quantiles problem
- Climate forecast with quantiles
- Energy load forecast
- Heteroscedasticity detection

### `generate_all_illustrations.py`

Convenience script that runs all illustration generators.

## Requirements

Install required dependencies:

```bash
uv add altair pandas numpy
uv add vl-convert-python  # For saving Altair charts as PNG
```

## Usage

### Generate all illustrations

```bash
python docs/scripts/generate_all_illustrations.py
```

### Generate specific documentation illustrations

```bash
# General bootstrapping illustrations
python docs/scripts/generate_bootstrap_illustrations.py

# Residual bootstrapping illustrations
python docs/scripts/generate_residual_bootstrap_illustrations.py
```

## Output

All generated images are saved to `docs/images/` in PNG format with descriptive filenames.

## Updating Documentation

After generating new illustrations:

1. The markdown files should reference images using relative paths:
   ```markdown
   ![Description](images/illustration_name.png)
   ```

2. Preview the documentation:
   ```bash
   open -a "Marked 2" docs/bootstrapping-for-time-series.md
   open -a "Marked 2" docs/residual-bootstrapping-time-series.md
   ```

## Customization

To modify an illustration:

1. Edit the corresponding function in the script
2. Run the script to regenerate the image
3. Preview the updated documentation

## Image Naming Convention

Images follow the pattern: `{concept}_{type}.png`

Examples:
- `original_data_fitted_model.png`
- `residual_acf_plot.png`
- `block_bootstrap_illustration.png`
- `forecast_intervals.png`

## Notes

- All scripts use `np.random.seed(42)` for reproducibility
- Images are saved at 600x300 or 600x400 pixels for optimal markdown display
- Color scheme: steelblue (primary), orange (secondary), red (emphasis)
- Follows CLAUDE.md guidelines: no emojis, clear naming, descriptive titles
