"""This file contains tests for the FixedComposite masker."""

import tempfile

import numpy as np
import pytest

import shap


def test_fixed_composite_init():
    """Test FixedComposite masker initialization stores the underlying masker."""
    masker = shap.maskers.Fixed()
    fc = shap.maskers.FixedComposite(masker)

    assert fc.masker is masker


def test_fixed_composite_attribute_propagation():
    """Test that attributes from the underlying masker are propagated."""
    masker = shap.maskers.Fixed()
    fc = shap.maskers.FixedComposite(masker)

    # Fixed masker has shape and clustering, so those should be propagated
    assert hasattr(fc, "shape")
    assert fc.shape == masker.shape
    assert hasattr(fc, "clustering")
    np.testing.assert_array_equal(fc.clustering, masker.clustering)


def test_fixed_composite_attribute_propagation_text_data():
    """Test that text_data flag is propagated from the underlying masker."""

    class TextMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)
            self.text_data = True

        def __call__(self, mask, x):
            return ([x],)

    masker = TextMasker()
    fc = shap.maskers.FixedComposite(masker)

    assert hasattr(fc, "text_data")
    assert fc.text_data is True


def test_fixed_composite_attribute_propagation_image_data():
    """Test that image_data flag is propagated from the underlying masker."""

    class ImageMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)
            self.image_data = True

        def __call__(self, mask, x):
            return ([x],)

    masker = ImageMasker()
    fc = shap.maskers.FixedComposite(masker)

    assert hasattr(fc, "image_data")
    assert fc.image_data is True


def test_fixed_composite_none_attributes_not_propagated():
    """Test that attributes set to None on the underlying masker are not propagated."""

    class MinimalMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)
            self.invariants = None
            self.feature_names = None

        def __call__(self, mask, x):
            return ([x],)

    masker = MinimalMasker()
    fc = shap.maskers.FixedComposite(masker)

    assert hasattr(fc, "shape")
    # None-valued attributes should NOT be propagated
    assert not hasattr(fc, "invariants")
    assert not hasattr(fc, "feature_names")


def test_fixed_composite_call_returns_masked_plus_original():
    """Test __call__ returns masked output concatenated with original args."""
    masker = shap.maskers.Fixed()
    fc = shap.maskers.FixedComposite(masker)

    test_input = np.array([1, 2, 3])
    mask = np.array([], dtype=bool)

    result = fc(mask, test_input)

    assert isinstance(result, tuple)
    # Fixed masker returns 1 element, plus 1 original arg wrapped in np.array
    assert len(result) == 2
    np.testing.assert_array_equal(result[0][0], test_input)
    np.testing.assert_array_equal(result[1][0], test_input)


def test_fixed_composite_call_wraps_non_tuple_masker_output():
    """Test __call__ wraps masker output in a tuple when masker returns a non-tuple."""

    class ArrayMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 3)

        def __call__(self, mask, x):
            # Return a plain array, not a tuple
            return x * mask

    masker = ArrayMasker()
    fc = shap.maskers.FixedComposite(masker)

    test_input = np.array([10, 20, 30])
    mask = np.array([True, False, True])

    result = fc(mask, test_input)

    assert isinstance(result, tuple)
    # 1 from wrapped masker output + 1 original arg
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], np.array([10, 0, 30]))
    np.testing.assert_array_equal(result[1][0], test_input)


def test_fixed_composite_call_with_multiple_args():
    """Test __call__ with multiple input arguments."""

    class TwoArgMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (None, 0)

        def __call__(self, mask, x, y):
            return ([x], [y])

    masker = TwoArgMasker()
    fc = shap.maskers.FixedComposite(masker)

    arg1 = np.array([1, 2])
    arg2 = np.array([3, 4])
    mask = np.array([], dtype=bool)

    result = fc(mask, arg1, arg2)

    assert isinstance(result, tuple)
    # 2 from masker output + 2 original args wrapped
    assert len(result) == 4
    np.testing.assert_array_equal(result[0][0], arg1)
    np.testing.assert_array_equal(result[1][0], arg2)
    np.testing.assert_array_equal(result[2][0], arg1)
    np.testing.assert_array_equal(result[3][0], arg2)


def test_fixed_composite_call_with_scalar_arg():
    """Test __call__ wraps scalar arguments correctly."""
    masker = shap.maskers.Fixed()
    fc = shap.maskers.FixedComposite(masker)

    mask = np.array([], dtype=bool)
    result = fc(mask, 42)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0][0] == 42
    assert result[1][0] == 42


def test_fixed_composite_serialization_roundtrip():
    """Test that FixedComposite can be serialized and deserialized."""
    masker = shap.maskers.Fixed()
    original = shap.maskers.FixedComposite(masker)

    with tempfile.TemporaryFile() as f:
        original.save(f)
        f.seek(0)
        loaded = shap.maskers.FixedComposite.load(f)

    # Verify the loaded masker produces the same output
    test_input = np.array([1, 2, 3])
    mask = np.array([], dtype=bool)

    original_result = original(mask, test_input)
    loaded_result = loaded(mask, test_input)

    assert len(original_result) == len(loaded_result)
    for orig, load in zip(original_result, loaded_result):
        np.testing.assert_array_equal(orig, load)


@pytest.mark.skip(
    reason="fails on travis and I don't know why yet...Ryan might need to take a look since this API will change soon anyway"
)
def test_fixed_composite_masker_call():
    """Test to make sure the FixedComposite masker works when masking everything."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    args = ("This is a test statement for fixed composite masker",)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    masker = shap.maskers.Text(tokenizer)
    mask = np.zeros(masker.shape(*args)[1], dtype=bool)

    fixed_composite_masker = shap.maskers.FixedComposite(masker)

    expected_fixed_composite_masked_output = (
        np.array([""]),
        np.array(["This is a test statement for fixed composite masker"]),
    )
    fixed_composite_masked_output = fixed_composite_masker(mask, *args)

    assert fixed_composite_masked_output == expected_fixed_composite_masked_output


def test_serialization_fixedcomposite_masker():
    """Make sure fixedcomposite serialization works."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", use_fast=False)
    underlying_masker = shap.maskers.Text(tokenizer)
    original_masker = shap.maskers.FixedComposite(underlying_masker)

    with tempfile.TemporaryFile() as temp_serialization_file:
        original_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_masker = shap.maskers.FixedComposite.load(temp_serialization_file)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output
