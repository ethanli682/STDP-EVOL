"""Checks for scientific aggregation, truncated logs, and heatmap inputs."""
import tempfile
from pathlib import Path
import unittest
import json
import numpy as np

from analyze_dottracing import (fitness, features, generation_summary,
                                parse_stdout, set_feature, bounds_from_yaml,
                                load_records, write_json)


class AnalysisTests(unittest.TestCase):
    def test_generation_statistics_pool_agents_and_average_replicates_once(self):
        rows = generation_summary({"seed 7": [
            {"epoch": 0, "iteration": 2, "fitnessScore": [-3, -1]},
            {"epoch": 0, "iteration": 2, "fitnessScore": 4},
            {"epoch": 0, "iteration": 3, "fitnessScore": -1},
            {"epoch": 1, "iteration": 0, "fitnessScore": 2},
            {"epoch": 1, "iteration": 0, "fitnessScore": None},
        ], "seed 9": [{"iteration": 2, "fitnessScore": -8}]})
        self.assertEqual(rows[0]["best_fitness"], 4)
        self.assertEqual(rows[0]["mean_fitness"], 1)
        self.assertEqual(rows[0]["n_individuals"], 2)
        self.assertEqual(rows[1]["best_fitness"], -1)  # not best-so-far
        self.assertEqual((rows[2]["epoch"], rows[2]["iteration"]), (1, 0))
        self.assertEqual(rows[3]["run"], "seed 9")

    def test_stdout_does_not_invent_generations_or_complete_partial_trials(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "worker.out"
            p.write_text("""---- Start Task -----\n[gene] gc_pc=1.000/2.000  pc_ac=1.500/0.500  ac_mc=0.900  inhib_ac=0.200  inhib_mc=-1.000
[gene] sparsity gc_pc=0.100/0.200  pc_ac=0.300/0.400  rec=0.500  ac_mc=0.600  mc_opp=[0.1, 0.2, 0.3, 0.4]
Training:
Episode 0: AC active 0.0%, MC active 0.0%
Testing:
Episode 0: AC active 45.0%, MC active 50.0%
RETURNED FITNESS: -0.01
---- Start Task -----\n[gene] gc_pc=3.000/3.000
""")
            records = parse_stdout(p)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ac_active_fraction"], .45)
        self.assertTrue(records[0]["ac_all_episodes_in_band"])
        self.assertEqual(features(records[0])["sparsity_mc_opp[3]"], .4)
        with self.assertRaisesRegex(ValueError, "missing iteration"):
            generation_summary({"seed": records})

    def test_nonfinite_results_are_not_individuals(self):
        for value in (None, float("inf"), float("nan"), [], [2, None]):
            self.assertIsNone(fitness({"fitnessScore": value}))

    def test_array_coordinates_change_only_selected_value(self):
        bounds, baseline = bounds_from_yaml(Path(__file__).with_name("task_dotTracing.yaml"))
        set_feature(baseline, "sparsity_mc_opp[2]", .75)
        self.assertEqual(baseline["sparsity_mc_opp"], [.5, .5, .75, .5])
        self.assertEqual(len(features({"param": baseline})), len(bounds))

    def test_framed_history_is_read_once_instead_of_duplicating_summary(self):
        from GA_utils_framed_records import FramedRecordWriter, framed_paths_from_csv
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "0.log.csv"
            p.write_text('{"iteration": 0, "fitnessScore": -2.5}\n')
            data, index = framed_paths_from_csv(str(p))
            record = {"iteration": 0, "fitnessScore": -2.5,
                      "param": {"scaler_gc_pc1": [1.2]}}
            with FramedRecordWriter(data, index) as writer:
                writer.append(record)
            loaded = load_records(d)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(features(loaded[0])["scaler_gc_pc1"], 1.2)

    def test_noninteger_generation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite integer"):
            generation_summary({"run": [{"iteration": .5, "fitnessScore": 1}]})

    def test_numpy_genes_from_framed_records_export_as_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "records.json"
            write_json(path, {"gene": np.array([1., 2.]), "score": np.float32(-2.5)})
            self.assertEqual(json.loads(path.read_text()), {"gene": [1., 2.], "score": -2.5})


if __name__ == "__main__":
    unittest.main()
