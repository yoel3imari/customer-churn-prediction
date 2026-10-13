import numpy as np
import pytest

from src.features.pipeline import create_preprocessing_pipeline, get_feature_names


@pytest.fixture
def full_df(sample_df):
    df = sample_df.copy()
    df["mourec"] = [300.0, 400.0, 350.0, 250.0]
    df["blckvce"] = [2, 3, 1, 4]
    df["unansvce"] = [5, 10, 7, 3]
    df["dropblk"] = [1, 2, 0, 3]
    df["income"] = [3, 5, 2, 4]
    df["setprc"] = [40.0, 50.0, 35.0, 60.0]
    return df


def test_create_preprocessing_pipeline_step_names():
    pipeline = create_preprocessing_pipeline()
    assert [name for name, _ in pipeline.steps] == ["clean", "features", "preprocessor"]


def test_create_preprocessing_pipeline_preprocessor_transformers():
    pipeline = create_preprocessing_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "scale" in transformer_names
    assert "ohe" in transformer_names


def test_get_feature_names_returns_list_of_strings(full_df):
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(full_df)
    names = get_feature_names(pipeline)
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    assert len(names) > 0


def test_get_feature_names_count_matches_transformed_output(full_df):
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(full_df)
    names = get_feature_names(pipeline)
    transformed = pipeline.named_steps["preprocessor"].transform(
        pipeline[:-1].transform(full_df)
    )
    assert len(names) == transformed.shape[1]


def test_full_pipeline_fit_transform_shape(full_df):
    pipeline = create_preprocessing_pipeline()
    result = pipeline.fit_transform(full_df)
    assert result.ndim == 2
    assert result.shape[0] == full_df.shape[0]


def test_full_pipeline_fit_transform_no_nan(full_df):
    pipeline = create_preprocessing_pipeline()
    result = pipeline.fit_transform(full_df)
    assert not np.isnan(result).any()
