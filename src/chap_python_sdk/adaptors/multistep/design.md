# Steps
- split the data into features and target (keeping location and time_period in both)
- (predict) join future and historic data (for X)
- Transform X in pipeline
- Transform y in prediction model (and inverse later). Prediction Model should take a tranformation pipeline as arguemnt 
- Model should handle converting to xarrays

# Needed changes
- MulitStepModel should have a subclass that fits and predict multiple regions. These mehtods should be called fit and predict
- MultiStepModel should accept dataframes and internally convert to xarray
- Transformation of X should be done using ColumnTransormer ala 

index_cols = ['time_period', 'location']
feature_cols = [c for c in X.columns if c not in index_cols]

ct = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), feature_cols)
    ],
    remainder='passthrough'
)
- Target transformations should be log transform and standard scaler in a pipeline that is passed to the init of the multistepmodel


