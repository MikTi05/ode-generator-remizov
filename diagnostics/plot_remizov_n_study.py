"""Plot persisted successful runs only; never call the solver."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/ode-remizov-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir', type=Path)
    args = parser.parse_args()
    folder = args.results_dir
    study = json.loads((folder / 'summary.json').read_text())
    grouped = defaultdict(list)
    for run in study['runs']:
        if run['status'] == 'success' and run['max_abs_error'] is not None:
            grouped[run['equation_id']].append(run)
    if not any(len(rows) >= 2 for rows in grouped.values()):
        raise SystemExit('Insufficient successful points for an error-vs-n curve')
    paths = [folder / f'error_vs_n.{extension}' for extension in ('png', 'svg')]
    if any(path.exists() for path in paths):
        raise SystemExit('Refusing to overwrite an existing figure')
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for equation_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row['n'])
        ax.plot([r['n'] for r in rows], [r['max_abs_error'] for r in rows],
                marker='o', linewidth=1.7, label=equation_id)
        for r in rows:
            ax.annotate(f"{r['max_abs_error']:.6f}",
                (r['n'], r['max_abs_error']), xytext=(7, 6),
                textcoords='offset points', fontsize=10)
    ax.set(xlabel='n (Chernoff operator applications per quadrature node)',
           ylabel='Max absolute error on the fixed 33-point grid',
           title='Successful runs only: two points do not establish convergence',
           xlim=(.8, 4.3), ylim=(0, 3.6), xticks=[1, 2, 4])
    ax.grid(alpha=.25)
    ax.legend(loc='upper right')
    ax.text(4, .15, 'n=4 not run\nafter series stop', ha='center',
            va='bottom', color='#666666', fontsize=9)
    for path in paths:
        fig.savefig(path, dpi=160)
    plt.close(fig)
    (folder/'plotting_environment.json').write_text(json.dumps({
        'matplotlib': matplotlib.__version__,
        'installed_versions': {d.metadata['Name']: d.version
            for d in importlib.metadata.distributions()},
        'source': 'summary.json; successful records only',
        'figure_files': [p.name for p in paths],
    }, indent=2, sort_keys=True) + '\n')
    print('\n'.join(str(path) for path in paths))


if __name__ == '__main__':
    main()
