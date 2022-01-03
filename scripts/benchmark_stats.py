import yaml
from pathlib import Path
import plot_stats


def main():
	benchmark_path = Path("../benchmark")
	results_path = Path("../results")
	tuning_path = Path("../tuning")

	# instances = ["carFirstOrder/bugtrap_0", "carFirstOrder/kink_0", "carFirstOrder/parallelpark_0"]
	# algs = ["sst", "sbpl", "dbAstar"]

	# instances = ["carFirstOrder/bugtrap_0", "carFirstOrder/kink_0", "carFirstOrder/parallelpark_0"]
	# algs = ["sst", "sbpl", "dbAstar-komo", "dbAstar-scp"]

	instances = ["carSecondOrder/parallelpark_0", "carSecondOrder/kink_0", "carSecondOrder/bugtrap_0"]
	algs = ["sst", "dbAstar-scp"]

	report = plot_stats.Report(results_path / "stats.pdf")

	for instance in instances:
		report.add("{}".format(instance))
		for alg in algs:
			result_folder = results_path / instance / alg
			stat_files = [str(p) for p in result_folder.glob("**/stats.yaml")]
			print(stat_files)
			if len(stat_files) > 0:
				report.load_stat_files(stat_files, 5*60, 0.1, alg)

	report.close()

if __name__ == '__main__':
	main()
