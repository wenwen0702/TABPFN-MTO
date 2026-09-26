"""Run the fixed paper method and save only final task values."""
import argparse
import csv
from pathlib import Path

PROBLEMS = ['CIHS', 'CIMS', 'CILS', 'PIHS', 'PIMS', 'PILS', 'NIHS', 'NIMS', 'NILS']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems', nargs='+', choices=PROBLEMS, default=PROBLEMS)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(11, 41)))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'final_results.csv')
    args = parser.parse_args(argv)
    if min(args.seeds) < 0:
        parser.error('Seeds must be non-negative.')
    if args.output.exists():
        parser.error('Output already exists. Choose a new --output filename.')
    from main import run_online_single_problem
    from CEC2017MTSO import BENCHMARK_FILES
    data_dir = Path(__file__).resolve().parent / 'Tasks'
    missing = [BENCHMARK_FILES[p] for p in args.problems if not (data_dir / BENCHMARK_FILES[p]).is_file()]
    if missing:
        parser.error('Missing Tasks/ files: ' + ', '.join(dict.fromkeys(missing)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Problem', 'Seed', 'BestA', 'BestB'])
        for problem in dict.fromkeys(args.problems):
            for seed in dict.fromkeys(args.seeds):
                result = run_online_single_problem(problem, init_seed=seed)
                writer.writerow([problem, seed, result['bestA'], result['bestB']])
                handle.flush()
                print(f"{problem}, seed={seed}: {result['bestA']:.6e}, {result['bestB']:.6e}")


if __name__ == '__main__':
    main()
