# House Price Predictor Notebook Design

## Purpose

Create a beginner-friendly Jupyter Notebook that teaches Tengku Ammer how a complete machine-learning regression workflow operates and lets him make house-price predictions without finding or downloading a dataset.

The notebook is an educational estimator. Its predictions must be described as simulated examples rather than real Malaysian property valuations.

## Deliverable

The main deliverable is `House_Price_Predictor.ipynb`. It must run from top to bottom in Jupyter Notebook and contain the explanations, generated dataset, trained model, evaluation, charts, and prediction interface in one file.

A short `README.md` will explain how to install the small dependency set, open the notebook, run every cell, and troubleshoot the most common beginner problems.

## User Experience

The learner opens the notebook and selects **Kernel > Restart & Run All**. The notebook then:

1. Imports the required libraries and fixes a random seed for reproducibility.
2. Generates a Malaysian-style educational housing dataset with prices in RM.
3. Displays sample rows and simple dataset summaries.
4. Separates input features from the target price and creates training and test sets.
5. Trains a preprocessing-and-regression pipeline.
6. Reports Mean Absolute Error, Root Mean Squared Error, and R² on unseen test data.
7. Displays actual-versus-predicted and feature-importance charts.
8. Shows an interactive prediction panel for entering a property's details.
9. Provides a plain Python prediction function as a fallback if notebook widgets are unavailable.

## Dataset

The notebook generates 2,000 fictional homes. Each row includes:

- `location_type`: City Centre, Suburban, or Small Town
- `property_type`: Apartment, Terrace, Semi-D, or Bungalow
- `floor_area_sqft`: positive floor area in square feet
- `bedrooms`: integer bedroom count
- `bathrooms`: integer bathroom count
- `property_age_years`: non-negative property age
- `distance_to_city_km`: non-negative distance to the city centre
- `parking_spaces`: integer parking-space count
- `price_rm`: simulated sale price in Malaysian ringgit

The target price will be generated from understandable relationships: more floor area, premium locations, larger property types, extra bathrooms, and parking increase price; age and distance generally reduce it. Random market noise prevents a perfectly deterministic result. Generated prices will be clipped to a sensible positive minimum.

The notebook must visibly state that the dataset is synthetic and must not imply that its coefficients represent the actual Malaysian housing market.

## Model Architecture

A scikit-learn `Pipeline` will combine preprocessing and a `RandomForestRegressor`:

- Categorical columns use `OneHotEncoder(handle_unknown="ignore")`.
- Numeric columns pass through unchanged.
- `ColumnTransformer` keeps preprocessing attached to the model.
- `RandomForestRegressor` learns nonlinear patterns without requiring the learner to understand advanced feature scaling.

All random operations use fixed seeds. The train/test split uses 80% of rows for training and 20% for final evaluation.

## Prediction Interface

The interactive panel will use `ipywidgets` controls and a clearly labelled **Predict Price** button. Inputs will be constrained to plausible ranges so accidental negative values cannot be submitted. The result will appear as a formatted RM amount and repeat the educational-use warning.

The notebook will also define `predict_house_price(...)`. A learner can change an example function call and run that cell even if widgets are not installed or enabled.

## Error Handling

- The first import cell will explain the exact installation command if a required package is missing.
- The widget section will catch widget-import failures and direct the learner to the fallback function.
- The prediction function will validate categorical choices and numeric ranges, raising plain-language `ValueError` messages for invalid data.
- The model is trained before either prediction path is displayed.

## Testing and Verification

Automated tests will verify that:

- The notebook is valid JSON and all code cells compile.
- Dataset generation returns 2,000 rows with the expected columns and no missing values.
- Prices and numeric house features stay within their defined bounds.
- The trained model returns finite, positive predictions.
- Evaluation metrics are finite and the test-set R² reaches at least 0.75 on the controlled synthetic data.
- The validation function rejects invalid categories and out-of-range numeric values.
- A clean, top-to-bottom notebook execution completes without errors in the available environment.

Visual output will also be inspected to ensure charts have readable titles, axes, and currency formatting.

## Scope Limits

The first version will not scrape property websites, call paid APIs, claim professional valuation accuracy, save user inputs, deploy a public website, or retrain on an uploaded CSV. These can be future improvements after the learner understands and uses the basic workflow.

