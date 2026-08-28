from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "House_Price_Predictor.ipynb"


def _stable_cell_id(kind: str, key: str, source: str, tags: tuple[str, ...]) -> str:
    identity = "\0".join((kind, key, source, ",".join(tags)))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{kind[0]}-{digest}"


def markdown(key: str, source: str):
    cleaned = source.strip()
    cell = nbf.v4.new_markdown_cell(cleaned, metadata={})
    cell["id"] = _stable_cell_id("markdown", key, cleaned, ())
    return cell


def code(key: str, source: str, *tags: str):
    cleaned = source.strip()
    stable_tags = tuple(tags)
    cell = nbf.v4.new_code_cell(
        cleaned,
        execution_count=None,
        outputs=[],
        metadata={"tags": list(stable_tags)},
    )
    cell["id"] = _stable_cell_id("code", key, cleaned, stable_tags)
    return cell


def build_notebook(output_path: Path = DEFAULT_OUTPUT) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }
    notebook.cells = [
        markdown(
            "title",
            """
# Malaysian House Price Predictor

**Educational demo only — not a real or professional property valuation.**

This notebook uses entirely fictional data. Its relationships and RM estimates are designed to teach machine learning; they do not represent the Malaysian property market.
""",
        ),
        markdown(
            "overview",
            """
## 1. What You Will Build

You will create 2,000 fictional homes, inspect the data, train a random-forest regression model, measure how it performs on unseen examples, and make a new prediction. Run the notebook from top to bottom with **Kernel > Restart & Run All** so each step has the variables produced by the step before it.

The first code cell imports every core dependency. If it reports a missing package, run the installation command it shows in a new cell, restart the kernel, and run all cells again.
""",
        ),
        code(
            "imports",
            """
try:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "A required package is missing. In a new notebook cell, run "
        "%pip install numpy pandas scikit-learn matplotlib ipywidgets, "
        "then restart the kernel and run all cells again."
    ) from error

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
plt.style.use("seaborn-v0_8-whitegrid")
""",
            "core-test",
            "import-guard",
        ),
        markdown(
            "dataset-introduction",
            """
## 2. Create the Educational Dataset

No file is downloaded. The function below generates Malaysian-style categories and plausible input ranges using a fixed random seed. The price formula rewards space, premium locations, larger property types, bathrooms, bedrooms, and parking; it subtracts age and distance effects and then adds random market noise. These coefficients are teaching devices, not claims about real prices.
""",
        ),
        code(
            "dataset-generation",
            """
LOCATION_TYPES = ("City Centre", "Suburban", "Small Town")
PROPERTY_TYPES = ("Apartment", "Terrace", "Semi-D", "Bungalow")


def generate_housing_data(n_rows=2000, random_state=42):
    '''Generate a reproducible, fictional housing dataset for education.'''
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 100:
        raise ValueError("n_rows must be an integer of at least 100")

    rng = np.random.default_rng(random_state)
    location = rng.choice(
        LOCATION_TYPES,
        n_rows,
        p=[0.30, 0.48, 0.22],
    )
    property_type = rng.choice(
        PROPERTY_TYPES,
        n_rows,
        p=[0.36, 0.38, 0.18, 0.08],
    )
    floor_area = np.clip(
        rng.normal(1750, 800, n_rows),
        450,
        6000,
    ).round().astype(int)
    bedrooms = np.clip(
        np.rint(floor_area / 550 + rng.normal(0.5, 0.8, n_rows)),
        1,
        8,
    ).astype(int)
    bathrooms = np.clip(
        np.rint(bedrooms * 0.65 + rng.normal(0.4, 0.6, n_rows)),
        1,
        7,
    ).astype(int)
    property_age = rng.integers(0, 61, n_rows)
    distance = np.round(rng.uniform(0.5, 70, n_rows), 1)
    parking = np.clip(rng.poisson(1.4, n_rows), 0, 5).astype(int)

    location_bonus = pd.Series(location).map(
        {"City Centre": 260_000, "Suburban": 100_000, "Small Town": 0}
    ).to_numpy()
    property_bonus = pd.Series(property_type).map(
        {"Apartment": 0, "Terrace": 80_000, "Semi-D": 230_000, "Bungalow": 520_000}
    ).to_numpy()
    price = (
        80_000
        + floor_area * 360
        + bedrooms * 16_000
        + bathrooms * 28_000
        + parking * 22_000
        + location_bonus
        + property_bonus
        - property_age * 3_800
        - distance * 4_200
        + rng.normal(0, 45_000, n_rows)
    )

    return pd.DataFrame(
        {
            "location_type": location,
            "property_type": property_type,
            "floor_area_sqft": floor_area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "property_age_years": property_age,
            "distance_to_city_km": distance,
            "parking_spaces": parking,
            "price_rm": np.clip(price, 100_000, None).round(2),
        }
    )


housing_data = generate_housing_data()
print(f"Generated {len(housing_data):,} fictional homes with random seed {RANDOM_SEED}.")
""",
            "core-test",
            "dataset-generation",
        ),
        markdown(
            "dataset-summary-explanation",
            """
Before training, inspect individual examples and broader patterns. The first table checks what one row looks like. The numeric summary reveals scale and bounds, while the location summary checks group sizes and simulated average prices.
""",
        ),
        code(
            "dataset-summary",
            """
from IPython.display import display

sample_rows = housing_data.head(5)
numeric_summary = (
    housing_data[
        [
            "floor_area_sqft",
            "bedrooms",
            "bathrooms",
            "property_age_years",
            "distance_to_city_km",
            "parking_spaces",
            "price_rm",
        ]
    ]
    .describe()
    .T[["min", "mean", "max"]]
    .round(2)
)
location_summary = (
    housing_data.groupby("location_type", observed=False)
    .agg(
        home_count=("price_rm", "size"),
        average_price_rm=("price_rm", "mean"),
    )
    .reindex(LOCATION_TYPES)
    .round({"average_price_rm": 2})
)

print("Sample fictional homes:")
display(sample_rows)
print("Useful numeric summary:")
display(numeric_summary)
print("Location summary:")
display(location_summary)
""",
            "core-test",
            "data-summary",
        ),
        markdown(
            "training-introduction",
            """
## 3. Train the Model

The target is `price_rm`; the other eight columns are inputs. An 80/20 split keeps 400 homes hidden for final evaluation. The pipeline one-hot encodes text categories and passes numeric columns through unchanged, keeping preprocessing and the random forest together.
""",
        ),
        code(
            "model-training",
            """
CATEGORICAL_FEATURES = ["location_type", "property_type"]
NUMERIC_FEATURES = [
    "floor_area_sqft",
    "bedrooms",
    "bathrooms",
    "property_age_years",
    "distance_to_city_km",
    "parking_spaces",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

X = housing_data[FEATURE_COLUMNS]
y = housing_data["price_rm"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

preprocessor = ColumnTransformer(
    [
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES,
        ),
        ("numbers", "passthrough", NUMERIC_FEATURES),
    ]
)
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
mae = float(mean_absolute_error(y_test, test_predictions))
rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))
r2 = float(r2_score(y_test, test_predictions))

print(f"Training homes: {len(X_train):,}")
print(f"Unseen test homes: {len(X_test):,}")
""",
            "core-test",
            "model-training",
        ),
        markdown(
            "evaluation-introduction",
            """
## 4. Evaluate the Model

- **MAE** is the typical absolute miss in RM.
- **RMSE** penalizes unusually large misses more strongly.
- **R²** measures how much variation the model explains; closer to 1 is better.

All three results below come from controlled fictional data, so they do not measure accuracy on real Malaysian homes.
""",
        ),
        code(
            "metrics-display",
            """
print(f"Mean Absolute Error (MAE): RM {mae:,.0f}")
print(f"Root Mean Squared Error (RMSE): RM {rmse:,.0f}")
print(f"R-squared (R²): {r2:.3f}")
print("These scores describe fictional controlled data, not real market accuracy.")
""",
            "core-test",
            "metrics-display",
        ),
        markdown(
            "actual-chart-explanation",
            """
Each dot compares a hidden home's simulated price with the model prediction. A dot on the dashed line would be exact. RM-formatted axes make the size of errors easier to read.
""",
        ),
        code(
            "actual-vs-predicted-figure",
            """
def format_rm_axis(value, position):
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"RM {value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"RM {value / 1_000:,.0f}k"
    return f"RM {value:,.0f}"


rm_axis_formatter = FuncFormatter(format_rm_axis)
actual_vs_predicted_figure, actual_vs_predicted_axis = plt.subplots(figsize=(8.5, 6))
actual_vs_predicted_axis.scatter(
    y_test,
    test_predictions,
    alpha=0.58,
    s=34,
    color="#176B87",
    edgecolors="none",
)
comparison_low = float(min(y_test.min(), test_predictions.min()))
comparison_high = float(max(y_test.max(), test_predictions.max()))
actual_vs_predicted_axis.plot(
    [comparison_low, comparison_high],
    [comparison_low, comparison_high],
    linestyle="--",
    linewidth=2,
    color="#C0392B",
    label="Ideal prediction",
)
actual_vs_predicted_axis.set_title("Actual vs Predicted House Prices")
actual_vs_predicted_axis.set_xlabel("Actual price (RM)")
actual_vs_predicted_axis.set_ylabel("Predicted price (RM)")
actual_vs_predicted_axis.xaxis.set_major_formatter(rm_axis_formatter)
actual_vs_predicted_axis.yaxis.set_major_formatter(rm_axis_formatter)
actual_vs_predicted_axis.legend()
actual_vs_predicted_figure.tight_layout()
plt.show()
""",
            "figure-test",
            "actual-vs-predicted",
        ),
        markdown(
            "importance-chart-explanation",
            """
Random forests assign a relative importance to every transformed input. The chart shows the ten largest values. Importance describes this simulated model; it does not prove that a feature causes real prices to change.
""",
        ),
        code(
            "feature-importance-figure",
            """
def readable_feature_name(transformed_name):
    location_prefix = "categories__location_type_"
    property_prefix = "categories__property_type_"
    if transformed_name.startswith(location_prefix):
        return "Location: " + transformed_name.removeprefix(location_prefix)
    if transformed_name.startswith(property_prefix):
        return "Property: " + transformed_name.removeprefix(property_prefix)
    plain_name = transformed_name.split("__", 1)[-1]
    return plain_name.replace("_", " ").title()


transformed_feature_names = model.named_steps["preprocessor"].get_feature_names_out()
feature_importance_table = pd.DataFrame(
    {
        "feature": [readable_feature_name(name) for name in transformed_feature_names],
        "importance": model.named_steps["regressor"].feature_importances_,
    }
).sort_values("importance", ascending=False, ignore_index=True)
top_ten_importances = feature_importance_table.head(10).sort_values("importance")

feature_importance_figure, feature_importance_axis = plt.subplots(figsize=(8.5, 6))
feature_importance_axis.barh(
    top_ten_importances["feature"],
    top_ten_importances["importance"],
    color="#4C956C",
)
feature_importance_axis.set_title("Most Influential Model Features")
feature_importance_axis.set_xlabel("Random forest importance")
feature_importance_axis.set_ylabel("")
feature_importance_figure.tight_layout()
plt.show()
""",
            "figure-test",
            "feature-importance",
        ),
        markdown(
            "prediction-introduction",
            """
## 5. Predict a House Price

Both prediction interfaces share the validation function below. It rejects unknown categories, booleans, text in numeric fields, NaN/infinity, values outside the generated-data ranges, and fractional counts. The function does not silently round an input.
""",
        ),
        code(
            "validation-and-prediction",
            """
NUMERIC_BOUNDS = {
    "floor_area_sqft": (450, 6000),
    "bedrooms": (1, 8),
    "bathrooms": (1, 7),
    "property_age_years": (0, 60),
    "distance_to_city_km": (0.5, 70),
    "parking_spaces": (0, 5),
}
COUNT_FEATURES = {
    "bedrooms",
    "bathrooms",
    "property_age_years",
    "parking_spaces",
}


def validate_house_inputs(
    location_type,
    property_type,
    floor_area_sqft,
    bedrooms,
    bathrooms,
    property_age_years,
    distance_to_city_km,
    parking_spaces,
):
    '''Validate one fictional home without coercing or rounding its values.'''
    if location_type not in LOCATION_TYPES:
        raise ValueError(f"location_type must be one of {LOCATION_TYPES}")
    if property_type not in PROPERTY_TYPES:
        raise ValueError(f"property_type must be one of {PROPERTY_TYPES}")

    numeric_values = {
        "floor_area_sqft": floor_area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_age_years": property_age_years,
        "distance_to_city_km": distance_to_city_km,
        "parking_spaces": parking_spaces,
    }
    accepted_numeric_types = (int, float, np.integer, np.floating)
    for name, value in numeric_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, accepted_numeric_types):
            raise ValueError(f"{name} must be a number")
        lower, upper = NUMERIC_BOUNDS[name]
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        if name in COUNT_FEATURES and not np.equal(value, np.floor(value)):
            raise ValueError(f"{name} must be a whole number")

    return {
        "location_type": location_type,
        "property_type": property_type,
        **numeric_values,
    }


def predict_house_price(
    location_type,
    property_type,
    floor_area_sqft,
    bedrooms,
    bathrooms,
    property_age_years,
    distance_to_city_km,
    parking_spaces,
):
    '''Return one simulated RM estimate for learning, not professional valuation.'''
    validated = validate_house_inputs(
        location_type,
        property_type,
        floor_area_sqft,
        bedrooms,
        bathrooms,
        property_age_years,
        distance_to_city_km,
        parking_spaces,
    )
    input_row = pd.DataFrame([validated], columns=FEATURE_COLUMNS)
    return float(model.predict(input_row)[0])
""",
            "core-test",
            "prediction-api",
        ),
        markdown(
            "fallback-explanation",
            """
The plain Python call below always works after training, even when notebook widgets are unavailable. Edit the values and rerun the cell. Keep each field within the ranges used by the fictional dataset.
""",
        ),
        code(
            "fallback-example",
            """
example_price = predict_house_price(
    location_type="Suburban",
    property_type="Terrace",
    floor_area_sqft=1800,
    bedrooms=4,
    bathrooms=3,
    property_age_years=8,
    distance_to_city_km=18,
    parking_spaces=2,
)
print(f"Predicted price: RM {example_price:,.0f}")
print("Educational estimate only — not a professional property valuation.")
""",
            "fallback-example",
        ),
        markdown(
            "interactive-predictor",
            """
## 6. Interactive Predictor

Choose a fictional home's details and click **Predict Price**. The controls use the same categories and bounds as the shared validation function. If controls do not appear, use `predict_house_price(...)` in the previous section.
""",
        ),
        code(
            "widget-interface",
            """
widgets_available = False
widget_last_message = ""

try:
    import ipywidgets as widgets
    from IPython.display import display as display_widget
except ImportError:
    print("Widgets are unavailable. Use predict_house_price(...) in the previous section.")
else:
    widgets_available = True
    widget_controls = {
        "location_type": widgets.Dropdown(
            options=LOCATION_TYPES,
            value="Suburban",
            description="Location:",
            style={"description_width": "150px"},
        ),
        "property_type": widgets.Dropdown(
            options=PROPERTY_TYPES,
            value="Terrace",
            description="Property type:",
            style={"description_width": "150px"},
        ),
        "floor_area_sqft": widgets.BoundedFloatText(
            value=1800,
            min=NUMERIC_BOUNDS["floor_area_sqft"][0],
            max=NUMERIC_BOUNDS["floor_area_sqft"][1],
            step=25,
            description="Floor area (sq ft):",
            style={"description_width": "150px"},
        ),
        "bedrooms": widgets.BoundedIntText(
            value=4,
            min=NUMERIC_BOUNDS["bedrooms"][0],
            max=NUMERIC_BOUNDS["bedrooms"][1],
            description="Bedrooms:",
            style={"description_width": "150px"},
        ),
        "bathrooms": widgets.BoundedIntText(
            value=3,
            min=NUMERIC_BOUNDS["bathrooms"][0],
            max=NUMERIC_BOUNDS["bathrooms"][1],
            description="Bathrooms:",
            style={"description_width": "150px"},
        ),
        "property_age_years": widgets.BoundedIntText(
            value=8,
            min=NUMERIC_BOUNDS["property_age_years"][0],
            max=NUMERIC_BOUNDS["property_age_years"][1],
            description="Age (years):",
            style={"description_width": "150px"},
        ),
        "distance_to_city_km": widgets.BoundedFloatText(
            value=18,
            min=NUMERIC_BOUNDS["distance_to_city_km"][0],
            max=NUMERIC_BOUNDS["distance_to_city_km"][1],
            step=0.5,
            description="Distance (km):",
            style={"description_width": "150px"},
        ),
        "parking_spaces": widgets.BoundedIntText(
            value=2,
            min=NUMERIC_BOUNDS["parking_spaces"][0],
            max=NUMERIC_BOUNDS["parking_spaces"][1],
            description="Parking spaces:",
            style={"description_width": "150px"},
        ),
    }
    predict_button = widgets.Button(
        description="Predict Price",
        button_style="success",
        icon="home",
    )
    prediction_output = widgets.Output()

    def _handle_predict_click(_button):
        global widget_last_message
        submitted = {name: control.value for name, control in widget_controls.items()}
        try:
            predicted_price = predict_house_price(**submitted)
            widget_last_message = (
                f"Predicted price: RM {predicted_price:,.0f}\\n"
                "Educational estimate only — not a professional property valuation."
            )
        except ValueError as error:
            widget_last_message = f"Please correct the input: {error}"
        with prediction_output:
            prediction_output.clear_output(wait=True)
            print(widget_last_message)

    predict_button.on_click(_handle_predict_click)
    widget_panel = widgets.VBox(
        [
            widgets.HTML("<b>Fictional house details</b>"),
            *widget_controls.values(),
            predict_button,
            prediction_output,
        ]
    )
    display_widget(widget_panel)
""",
            "widget-test",
            "interactive-predictor",
        ),
        markdown(
            "exercises",
            """
## 7. What to Try Next

Try these beginner exercises after the notebook runs successfully:

1. **Compare locations:** Keep one house unchanged and predict it in City Centre, Suburban, and Small Town. Which simulated estimate is largest, and how does that relate to the generator?
2. **Test age:** Compare the same fictional house at 0 and 40 years old. Explain the direction of the change without treating it as real market advice.
3. **Change the seed:** Use `generate_housing_data(random_state=7)`, retrain, and compare R². Why does reproducible randomness matter?
4. **Add a category:** Add a fictional location and bonus, then update `LOCATION_TYPES`. Notice how shared constants keep generation, validation, and widgets aligned.

Remember: every result in this notebook is an educational simulation, not a professional valuation.
""",
        ),
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    build_notebook(destination)
