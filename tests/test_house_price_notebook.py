from __future__ import annotations

import builtins
import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest
from ipykernel.inprocess.blocking import BlockingInProcessKernelClient
from ipykernel.inprocess.manager import InProcessKernelManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "House_Price_Predictor.ipynb"
BUILD_SCRIPT = ROOT / "scripts" / "build_notebook.py"
EXPECTED_COLUMNS = [
    "location_type",
    "property_type",
    "floor_area_sqft",
    "bedrooms",
    "bathrooms",
    "property_age_years",
    "distance_to_city_km",
    "parking_spaces",
    "price_rm",
]


def load_notebook(path: Path = NOTEBOOK_PATH):
    return nbformat.read(path, as_version=4)


def cells_with_tag(notebook, tag: str):
    return [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and tag in cell.metadata.get("tags", [])
    ]


def one_cell_with_tag(notebook, tag: str):
    matches = cells_with_tag(notebook, tag)
    assert len(matches) == 1, f"expected one {tag!r} cell, found {len(matches)}"
    return matches[0]


def guarded_builtins(blocked_package: str):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == blocked_package or name.startswith(f"{blocked_package}."):
            raise ModuleNotFoundError(f"No module named {blocked_package!r}")
        return real_import(name, globals, locals, fromlist, level)

    namespace = vars(builtins).copy()
    namespace["__import__"] = guarded_import
    return namespace


@pytest.fixture(scope="session")
def notebook():
    return load_notebook()


@pytest.fixture(scope="session")
def core_namespace(notebook):
    namespace = {"__name__": "notebook_core_test"}
    for cell in cells_with_tag(notebook, "core-test"):
        exec(compile(cell.source, NOTEBOOK_PATH.name, "exec"), namespace)
    return namespace


@pytest.fixture(scope="session")
def figure_namespace(notebook, core_namespace):
    namespace = dict(core_namespace)
    for cell in cells_with_tag(notebook, "figure-test"):
        exec(compile(cell.source, NOTEBOOK_PATH.name, "exec"), namespace)
    return namespace


class NbClientInProcessKernelClient(BlockingInProcessKernelClient):
    """Adapt ipykernel's queue client to nbclient's timeout signature."""

    def wait_for_ready(self, timeout=None):
        return super().wait_for_ready()


class NbClientInProcessKernelManager(InProcessKernelManager):
    """Run a real IPython kernel without ZeroMQ sockets."""

    def client(self, **kwargs):
        kwargs["kernel"] = self.kernel
        return NbClientInProcessKernelClient(
            parent=self,
            session=self.session,
            **kwargs,
        )

    def shutdown_kernel(self, now=False, restart=False):
        return super().shutdown_kernel()


def test_independent_builds_are_byte_identical_with_valid_unique_cell_ids(tmp_path):
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"
    for destination in (first, second):
        result = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), str(destination)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    assert first.read_bytes() == second.read_bytes()
    cell_ids = [cell.id for cell in load_notebook(first).cells]
    assert len(cell_ids) == len(set(cell_ids))
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) for cell_id in cell_ids)


def test_notebook_is_valid_json_v4_and_all_code_cells_compile(notebook):
    raw_notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    nbformat.validate(raw_notebook)
    assert notebook.nbformat == 4
    assert len(cells_with_tag(notebook, "core-test")) >= 4
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, NOTEBOOK_PATH.name, "exec")


def test_first_guarded_cell_loads_the_complete_core_dependency_surface(notebook):
    import_cell = one_cell_with_tag(notebook, "import-guard")
    assert next(cell for cell in notebook.cells if cell.cell_type == "code") is import_cell
    namespace = {}
    exec(compile(import_cell.source, NOTEBOOK_PATH.name, "exec"), namespace)
    for required_name in (
        "np",
        "pd",
        "ColumnTransformer",
        "OneHotEncoder",
        "train_test_split",
        "mean_absolute_error",
        "mean_squared_error",
        "r2_score",
        "RandomForestRegressor",
        "Pipeline",
        "matplotlib",
        "plt",
        "FuncFormatter",
    ):
        assert required_name in namespace


def test_guarded_core_imports_give_friendly_guidance_for_sklearn_and_matplotlib(notebook):
    import_cell = one_cell_with_tag(notebook, "import-guard")
    for blocked_package in ("sklearn", "matplotlib"):
        namespace = {"__builtins__": guarded_builtins(blocked_package)}
        with pytest.raises(ModuleNotFoundError) as error:
            exec(compile(import_cell.source, NOTEBOOK_PATH.name, "exec"), namespace)
        message = str(error.value)
        assert "A required package is missing" in message
        assert "%pip install numpy pandas scikit-learn matplotlib ipywidgets" in message
        assert "restart the kernel" in message.lower()


def test_default_dataset_matches_the_approved_seed_and_formula(core_namespace):
    generated = core_namespace["generate_housing_data"]()
    assert generated.shape == (2000, 9)
    assert generated.columns.tolist() == EXPECTED_COLUMNS
    assert not generated.isna().any().any()
    assert generated.iloc[:3].to_dict("records") == [
        {
            "location_type": "Suburban",
            "property_type": "Semi-D",
            "floor_area_sqft": 918,
            "bedrooms": 3,
            "bathrooms": 2,
            "property_age_years": 41,
            "distance_to_city_km": 50.3,
            "parking_spaces": 0,
            "price_rm": 533370.37,
        },
        {
            "location_type": "Suburban",
            "property_type": "Terrace",
            "floor_area_sqft": 553,
            "bedrooms": 3,
            "bathrooms": 1,
            "property_age_years": 19,
            "distance_to_city_km": 2.3,
            "parking_spaces": 2,
            "price_rm": 461537.06,
        },
        {
            "location_type": "Small Town",
            "property_type": "Bungalow",
            "floor_area_sqft": 2832,
            "bedrooms": 7,
            "bathrooms": 5,
            "property_age_years": 54,
            "distance_to_city_km": 10.4,
            "parking_spaces": 1,
            "price_rm": 1626201.78,
        },
    ]
    assert generated["price_rm"].sum() == pytest.approx(1_742_718_424.07)


def test_data_generation_is_reproducible_for_any_fixed_seed(core_namespace):
    generate = core_namespace["generate_housing_data"]
    first = generate(n_rows=150, random_state=7)
    second = generate(n_rows=150, random_state=7)
    different = generate(n_rows=150, random_state=8)
    assert first.equals(second)
    assert not first.equals(different)


def test_all_generated_categories_and_numeric_ranges_are_valid(core_namespace):
    data = core_namespace["housing_data"]
    assert set(data["location_type"]) == {"City Centre", "Suburban", "Small Town"}
    assert set(data["property_type"]) == {
        "Apartment",
        "Terrace",
        "Semi-D",
        "Bungalow",
    }
    expected_bounds = {
        "floor_area_sqft": (450, 6000),
        "bedrooms": (1, 8),
        "bathrooms": (1, 7),
        "property_age_years": (0, 60),
        "distance_to_city_km": (0.5, 70),
        "parking_spaces": (0, 5),
    }
    for column, (lower, upper) in expected_bounds.items():
        assert data[column].between(lower, upper, inclusive="both").all()
    assert (data["price_rm"] >= 100_000).all()
    for column in ("floor_area_sqft", "bedrooms", "bathrooms", "property_age_years", "parking_spaces"):
        assert np.equal(data[column], np.floor(data[column])).all()


def test_data_generator_rejects_invalid_row_counts(core_namespace):
    generate = core_namespace["generate_housing_data"]
    for invalid in (True, 99, 200.0, "2000"):
        with pytest.raises(ValueError, match="integer of at least 100"):
            generate(invalid)


def test_useful_sample_numeric_and_location_summaries_precede_training(notebook, core_namespace):
    summary_cell = one_cell_with_tag(notebook, "data-summary")
    training_cell = one_cell_with_tag(notebook, "model-training")
    assert notebook.cells.index(summary_cell) < notebook.cells.index(training_cell)

    sample_rows = core_namespace["sample_rows"]
    numeric_summary = core_namespace["numeric_summary"]
    location_summary = core_namespace["location_summary"]
    assert sample_rows.shape == (5, 9)
    assert {"min", "mean", "max"}.issubset(numeric_summary.columns)
    assert {"floor_area_sqft", "distance_to_city_km", "price_rm"}.issubset(
        numeric_summary.index
    )
    assert list(location_summary.index) == ["City Centre", "Suburban", "Small Town"]
    assert location_summary["home_count"].sum() == 2000
    assert (location_summary["average_price_rm"] > 100_000).all()


def test_pipeline_and_split_match_the_approved_architecture(core_namespace):
    model = core_namespace["model"]
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    encoder = preprocessor.named_transformers_["categories"]
    assert encoder.handle_unknown == "ignore"
    assert regressor.n_estimators == 250
    assert regressor.random_state == 42
    assert regressor.n_jobs == -1
    assert len(core_namespace["X_train"]) == 1600
    assert len(core_namespace["X_test"]) == 400
    assert core_namespace["X_train"].index[:5].tolist() == [968, 240, 819, 692, 420]
    assert core_namespace["X_test"].index[:5].tolist() == [1860, 353, 1333, 905, 1289]


def test_evaluation_metrics_are_finite_and_r2_meets_quality_floor(core_namespace):
    for metric_name in ("mae", "rmse", "r2"):
        assert np.isfinite(core_namespace[metric_name])
    assert core_namespace["mae"] > 0
    assert core_namespace["rmse"] >= core_namespace["mae"]
    assert core_namespace["r2"] >= 0.75


def test_plain_prediction_returns_a_positive_finite_float(core_namespace):
    prediction = core_namespace["predict_house_price"](
        location_type="Suburban",
        property_type="Terrace",
        floor_area_sqft=1800,
        bedrooms=4,
        bathrooms=3,
        property_age_years=8,
        distance_to_city_km=18,
        parking_spaces=2,
    )
    assert isinstance(prediction, float)
    assert np.isfinite(prediction)
    assert prediction > 0


def test_validation_rejects_unknown_categories(core_namespace):
    validate = core_namespace["validate_house_inputs"]
    defaults = dict(
        location_type="Suburban",
        property_type="Terrace",
        floor_area_sqft=1800,
        bedrooms=4,
        bathrooms=3,
        property_age_years=8,
        distance_to_city_km=18,
        parking_spaces=2,
    )
    for field, invalid in (("location_type", "Moon"), ("property_type", "Castle")):
        values = defaults | {field: invalid}
        with pytest.raises(ValueError, match=rf"{field} must be one of"):
            validate(**values)


def test_validation_rejects_booleans_and_non_numeric_values(core_namespace):
    validate = core_namespace["validate_house_inputs"]
    defaults = dict(
        location_type="Suburban",
        property_type="Terrace",
        floor_area_sqft=1800,
        bedrooms=4,
        bathrooms=3,
        property_age_years=8,
        distance_to_city_km=18,
        parking_spaces=2,
    )
    for field, invalid in (
        ("floor_area_sqft", True),
        ("bedrooms", False),
        ("bathrooms", "3"),
        ("distance_to_city_km", object()),
    ):
        with pytest.raises(ValueError, match=rf"{field} must be a number"):
            validate(**(defaults | {field: invalid}))


def test_validation_rejects_non_finite_and_out_of_range_values(core_namespace):
    validate = core_namespace["validate_house_inputs"]
    defaults = dict(
        location_type="Suburban",
        property_type="Terrace",
        floor_area_sqft=1800,
        bedrooms=4,
        bathrooms=3,
        property_age_years=8,
        distance_to_city_km=18,
        parking_spaces=2,
    )
    for field, invalid in (
        ("floor_area_sqft", np.nan),
        ("floor_area_sqft", np.inf),
        ("bedrooms", 0),
        ("bathrooms", 8),
        ("property_age_years", -1),
        ("distance_to_city_km", 70.1),
        ("parking_spaces", 6),
    ):
        with pytest.raises(ValueError, match=rf"{field} must be between"):
            validate(**(defaults | {field: invalid}))


def test_validation_rejects_every_fractional_count_without_precision_narrowing(core_namespace):
    validate = core_namespace["validate_house_inputs"]
    defaults = dict(
        location_type="Suburban",
        property_type="Terrace",
        floor_area_sqft=1800,
        bedrooms=4,
        bathrooms=3,
        property_age_years=8,
        distance_to_city_km=18,
        parking_spaces=2,
    )
    fractional_counts = {
        "bedrooms": np.longdouble("4.0000000000000001"),
        "bathrooms": 3.5,
        "property_age_years": np.float64(8.25),
        "parking_spaces": np.longdouble("2.0000000000000001"),
    }
    for field, invalid in fractional_counts.items():
        with pytest.raises(ValueError, match=rf"{field} must be a whole number"):
            validate(**(defaults | {field: invalid}))


def test_validation_accepts_in_range_numpy_scalars_and_integral_float_counts(core_namespace):
    validated = core_namespace["validate_house_inputs"](
        location_type="City Centre",
        property_type="Apartment",
        floor_area_sqft=np.float64(1200.5),
        bedrooms=np.int64(3),
        bathrooms=np.float64(2.0),
        property_age_years=np.longdouble("4.0"),
        distance_to_city_km=np.longdouble("12.25"),
        parking_spaces=np.int32(1),
    )
    assert validated["floor_area_sqft"] == np.float64(1200.5)
    assert validated["distance_to_city_km"] == np.longdouble("12.25")


def test_actual_vs_predicted_figure_has_rm_axes_and_an_ideal_reference(figure_namespace):
    figure = figure_namespace["actual_vs_predicted_figure"]
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert axis.get_title() == "Actual vs Predicted House Prices"
    assert axis.get_xlabel() == "Actual price (RM)"
    assert axis.get_ylabel() == "Predicted price (RM)"
    assert len(axis.collections) == 1
    assert len(axis.lines) == 1
    assert axis.xaxis.get_major_formatter()(500_000, 0).startswith("RM")
    assert axis.yaxis.get_major_formatter()(1_000_000, 0).startswith("RM")


def test_feature_importance_figure_shows_ten_readable_features(figure_namespace):
    figure = figure_namespace["feature_importance_figure"]
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert axis.get_title() == "Most Influential Model Features"
    assert axis.get_xlabel() == "Random forest importance"
    assert len(axis.patches) == 10
    labels = [tick.get_text() for tick in axis.get_yticklabels()]
    assert len(labels) == 10
    assert all(label and "__" not in label for label in labels)
    assert np.isclose(figure_namespace["feature_importance_table"]["importance"].sum(), 1.0)


def test_fallback_example_prints_price_and_educational_warning(notebook, core_namespace):
    fallback_cell = one_cell_with_tag(notebook, "fallback-example")
    namespace = dict(core_namespace)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(fallback_cell.source, NOTEBOOK_PATH.name, "exec"), namespace)
    printed = output.getvalue()
    assert namespace["example_price"] > 0
    assert "Predicted price: RM" in printed
    assert "Educational estimate only" in printed
    assert "not a professional property valuation" in printed


def test_widget_cell_has_optional_fallback_and_real_working_button_callback(notebook, core_namespace):
    widget_cell = one_cell_with_tag(notebook, "widget-test")

    missing_namespace = dict(core_namespace)
    missing_namespace["__builtins__"] = guarded_builtins("ipywidgets")
    missing_output = io.StringIO()
    with contextlib.redirect_stdout(missing_output):
        exec(compile(widget_cell.source, NOTEBOOK_PATH.name, "exec"), missing_namespace)
    assert missing_namespace["widgets_available"] is False
    assert "Use predict_house_price" in missing_output.getvalue()

    namespace = dict(core_namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(widget_cell.source, NOTEBOOK_PATH.name, "exec"), namespace)
        namespace["predict_button"].click()
    assert namespace["widgets_available"] is True
    assert namespace["predict_button"].description == "Predict Price"
    assert namespace["widget_last_message"].startswith("Predicted price: RM")
    assert "Educational estimate only" in namespace["widget_last_message"]
    assert "professional property valuation" in namespace["widget_last_message"]


def test_notebook_executes_top_to_bottom_with_real_in_process_nbclient(tmp_path):
    notebook = load_notebook()
    manager = NbClientInProcessKernelManager()
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=240,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        executed = client.execute()
        namespace = manager.kernel.shell.user_ns
        assert namespace["housing_data"].shape == (2000, 9)
        assert namespace["r2"] >= 0.75
        assert callable(namespace["predict_house_price"])
        assert all(
            cell.execution_count is not None
            for cell in executed.cells
            if cell.cell_type == "code"
        )
        executed_path = tmp_path / "executed.ipynb"
        nbformat.write(executed, executed_path)
        assert executed_path.exists()
    finally:
        if manager.kernel is not None:
            manager.shutdown_kernel()
