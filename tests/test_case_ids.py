import pytest

from src.dataset.case_ids import normalize_case_id


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0001", "1"),
        ("rca_0508.npz", "508"),
        ("/data/lca_0023.npz", "23"),
        ("/data/lca/42/prefix_02.npz", "42"),
    ],
)
def test_imagecas_case_id_normalisation(value, expected):
    assert normalize_case_id(value, "imagecas_numeric") == expected


def test_imagecas_case_id_rejects_ambiguous_paths():
    with pytest.raises(ValueError, match="ImageCAS case IDs"):
        normalize_case_id("/data/lca/unknown/prefix_02.npz", "imagecas_numeric")


def test_literal_case_ids_are_unchanged():
    assert normalize_case_id("case-A", "literal") == "case-A"
