from pathlib import Path

from run import _clear_previous_results


def test_clear_previous_results_removes_only_generated_artifacts(tmp_path: Path):
    generated = (
        "estimation.qnr",
        "estimation.csv",
        "ground_truth.qnr",
        "ground_truth.json",
        "ingestion_report.json",
        "summary.json",
        "processed_accelerometer.csv",
        "processed_gyroscope.csv",
        "config.ini",
        "config_full.ini",
    )
    for name in generated:
        (tmp_path / name).write_text("old", encoding="utf-8")
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "position.png").write_bytes(b"old")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    _clear_previous_results(tmp_path)

    assert all(not (tmp_path / name).exists() for name in generated)
    assert not figures.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
