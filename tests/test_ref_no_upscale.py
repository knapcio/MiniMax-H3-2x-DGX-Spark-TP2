"""CPU tests of the H3_TP2_REF_NO_UPSCALE shape rule (h3_tp2/ref_shape.py)."""

import pytest

from h3_tp2.ref_shape import align, native_or_resized


def test_a_reference_at_or_above_the_threshold_follows_upstream():
    # the 1344x768 scene frame at a 1024 short edge, threshold 600: upstream's scaling and rounding to 32
    assert native_or_resized((1344, 768), 1024, 600) == (1792, 1024)
    assert native_or_resized((4096, 2304), 1024, 600) == (1824, 1024)
    assert native_or_resized((600, 1000), 1024, 600) == (1024, align(1000 * 1024 / 600))


def test_a_small_reference_keeps_its_own_size():
    # the 512 px character crops are not blown up to 1024: a quarter of the rows
    assert native_or_resized((512, 798), 1024, 600) == (512, 800)
    assert native_or_resized((512, 818), 1024, 600) == (512, 832)
    # threshold 0 = upstream's behaviour for every image
    assert native_or_resized((512, 798), 1024, 0) == (1024, align(798 * 2))


def test_alignment_and_bad_sizes():
    assert align(255) == 256 and align(16) == 32 and align(1000) == 992
    with pytest.raises(ValueError):
        native_or_resized((0, 100), 1024, 600)
