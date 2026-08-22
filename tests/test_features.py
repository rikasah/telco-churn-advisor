from features import BOOLEAN, CATEGORICAL, FEATURE_COLUMNS, NUMERIC


def test_categorical_and_numeric_do_not_overlap():
    assert set(CATEGORICAL).isdisjoint(set(NUMERIC))


def test_feature_columns_is_union_of_categorical_and_numeric():
    assert set(FEATURE_COLUMNS) == set(CATEGORICAL) | set(NUMERIC)


def test_boolean_columns_are_subset_of_numeric():
    assert set(BOOLEAN).issubset(set(NUMERIC))


def test_no_duplicate_feature_columns():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
