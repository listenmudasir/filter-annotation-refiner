"""Profiles must adapt to any dataset's geometry, not to known class names."""

from refiner.profiles import (
    DEFAULT_PROFILE,
    ClassProfile,
    ClassStats,
    derive_profile,
    derive_profiles,
    profile_for,
)


def boxes(w: float, h: float, n: int = 40):
    return [(0.0, 0.0, w, h)] * n


def profile(w: float, h: float, n: int = 40) -> ClassProfile:
    return derive_profile(ClassStats.from_boxes(0, "c", boxes(w, h, n)))


def test_thin_structures_disable_morphology():
    """Hair, scratches, cracks and fibres must not be closed or opened."""
    for w, h in [(900, 3), (200, 2), (3, 900), (600, 8)]:
        p = profile(w, h)
        assert p.preserve_thin, (w, h)
        assert p.close_kernel == 0
        assert p.open_kernel == 0
        assert p.target_fidelity >= 0.995


def test_compact_objects_get_morphology():
    for w, h in [(40, 38), (300, 160), (500, 480)]:
        p = profile(w, h)
        assert not p.preserve_thin, (w, h)
        assert p.close_kernel >= 3
        assert p.close_kernel % 2 == 1, "morphology kernels must be odd"


def test_tiny_objects_are_treated_as_thin_whatever_their_shape():
    """A 10x10 object cannot survive a 3x3 close regardless of aspect ratio."""
    p = profile(10, 10)
    assert p.preserve_thin
    assert p.close_kernel == 0


def test_speckle_floor_scales_with_object_size():
    small = profile(40, 38)
    medium = profile(300, 160)
    large = profile(900, 880)
    assert small.min_component_area < medium.min_component_area < large.min_component_area
    # A floor large enough to delete real geometry is never produced.
    assert large.min_component_area <= 256


def test_thin_classes_tolerate_more_leakage():
    assert profile(900, 3).max_leakage > profile(300, 300).max_leakage


def test_profiles_are_per_class_not_per_dataset():
    derived = derive_profiles(
        {0: boxes(900, 3), 1: boxes(300, 280)},
        {0: "fibre", 1: "blob"},
    )
    assert derived[0].preserve_thin
    assert not derived[1].preserve_thin


def test_identical_geometry_gives_identical_profiles_regardless_of_name():
    """The same shapes must convert the same way whatever the classes are called."""
    a = derive_profiles({0: boxes(900, 3)}, {0: "Hair"})
    b = derive_profiles({0: boxes(900, 3)}, {0: "wire_defect"})
    assert a[0].target_fidelity == b[0].target_fidelity
    assert a[0].preserve_thin == b[0].preserve_thin
    assert a[0].min_component_area == b[0].min_component_area


def test_unknown_class_falls_back_to_default():
    assert profile_for(99, {}, {}) is DEFAULT_PROFILE


def test_override_beats_derived():
    derived = derive_profiles({0: boxes(900, 3)}, {0: "fibre"})
    assert derived[0].preserve_thin
    override = ClassProfile(0.99, 4, 5, 0, False, 0.2)
    resolved = profile_for(0, derived, {0: override})
    assert not resolved.preserve_thin
    assert resolved.origin == "user override"


def test_empty_class_falls_back_without_crashing():
    p = derive_profile(ClassStats.from_boxes(0, "empty", []))
    assert p.min_component_area == DEFAULT_PROFILE.min_component_area
    assert "no samples" in p.origin


def test_origin_is_recorded_for_reporting():
    p = profile(900, 3)
    assert "thin" in p.origin and "n=40" in p.origin


def test_degenerate_boxes_do_not_divide_by_zero():
    p = derive_profile(ClassStats.from_boxes(0, "deg", [(5.0, 5.0, 5.0, 5.0)] * 5))
    assert p.min_component_area >= 2
    assert 0.0 < p.target_fidelity <= 1.0
