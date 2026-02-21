- Have the multistep model output pd.dataframe
- Create a transformer for adding location as a onehot encoded feature
- Create a transformer for adding season as a onehot encoded feature
- Create a tranformer for adding location x season as interaction
- Create a multistep model that make n_steps different models, removing one feature_lag, takes a callback function get_lag_idx to determine which columns to remove
- Create a deterministic multistep model that recursively predicts only mean/median/mode

Use the dhis2_eo Laos dataset to create a tutorial

Pages should be

- What are the assumptions of Linear Regression
- How are they broken in our type of time series models
- How can we allevieate the broken assumptions

- A simple multistep model, no lags, deterministic
- Introduce different effects one at a time
  - Seasonal (categorical and other)
  - Location
  - Lagged features
  - Lagged target
  - log transform
  - feature standardization
  - rate transform (divide by population)

- Show the abstractions used by multistep model:
  - sklearn onestep model
  - skpro wrapper to get uncertainty
  - mulitstep model
  - feature transformations
- Show how these concepts are used in different time series libraries

- Show a similar model framework in r
    