"""
Continuous Integration (CI) Test Suite for Sensor Telemetry Pipeline.
Executes 6 mandatory validation gates asserting temporal completeness, monotonicity,
zero missingness, jitter bounds, physical validity, and operational hurdle consistency.
"""

import unittest
import pandas as pd
from data_cleaning_pipeline import run_data_cleaning_pipeline


class TestSensorTelemetryCI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = run_data_cleaning_pipeline()

    def test_01_grid_completeness(self):
        """Assert timeline contains exactly 518,400 5-second steps across 30.0 days."""
        self.assertEqual(len(self.df), 518400)
        expected_start = pd.Timestamp("2025-09-09 00:00:00")
        expected_end = pd.Timestamp("2025-10-08 23:59:55")
        self.assertEqual(self.df.index.min(), expected_start)
        self.assertEqual(self.df.index.max(), expected_end)

    def test_02_monotonic_causality(self):
        """Assert strictly increasing timestamps without negative jumps or duplicates."""
        self.assertTrue(self.df.index.is_monotonic_increasing)
        time_diffs = self.df.index.to_series().diff().dropna()
        self.assertTrue((time_diffs == pd.Timedelta(seconds=5)).all())

    def test_03_zero_missingness_in_imputed_target(self):
        """Assert zero remaining NaN/Inf values in the target signal."""
        self.assertEqual(self.df["Reading_imputed"].isnull().sum(), 0)
        self.assertFalse(self.df["Reading_imputed"].isna().any())

    def test_04_phase_offset_boundedness(self):
        """Assert sampling jitter delta_t is strictly within [-2.5s, +2.5s]."""
        self.assertGreaterEqual(self.df["phase_offset_sec"].min(), -2.5)
        self.assertLessEqual(self.df["phase_offset_sec"].max(), 2.5)

    def test_05_physical_value_bounds(self):
        """Assert zero-clamping integrity (Reading >= 0.0)."""
        self.assertGreaterEqual(self.df["Reading_imputed"].min(), 0.0)

    def test_06_operational_hurdle_consistency(self):
        """Assert hurdle states conform to reading threshold (threshold = 0.05)."""
        dormant_mask = self.df["Reading_imputed"] <= 0.05
        self.assertTrue((self.df.loc[dormant_mask, "is_operational"] == 0).all())
        active_mask = self.df["Reading_imputed"] > 0.05
        self.assertTrue((self.df.loc[active_mask, "is_operational"] == 1).all())


def run_ci_suite() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSensorTelemetryCI)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ci_suite()
    if not success:
        exit(1)
