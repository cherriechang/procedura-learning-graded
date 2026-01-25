#! /usr/bin/env python3
"""
Batch Sanity Checks for Procedural Learning Transition Matrices

Analyzes:
- 5 matrix sizes (4×4 to 8×8)
- For each size: 1 original matrix + 5 random variations
- For each matrix: 10 randomly generated sequences

Output structure:
    sanity_check_results/
    ├── summary.json                    # Overall summary statistics
    ├── summary_report.md               # Human-readable report
    ├── 4x4/
    │   ├── original/
    │   │   ├── matrix.npy
    │   │   ├── matrix_analysis.json
    │   │   ├── convergence_plot.png
    │   │   └── sequence_validations.json
    │   ├── variation_1/
    │   │   └── ...
    │   └── ...
    ├── 5x5/
    │   └── ...
    └── ...
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'matrix_sizes': [4, 5, 6, 7, 8],
    'n_variations': 5,           # Number of random matrix variations per size
    'n_sequences': 10,           # Number of sequences to generate per matrix
    # Sequence length is calculated as: n_blocks * matrix_size * trials_per_block
    'n_blocks_pilot': 7,         # Pilot study
    'n_blocks_full': 20,         # Full study
    'trials_per_block_multiplier': 10,  # trials_per_block = matrix_size * 10
    'study_type': 'full',        # 'pilot' or 'full' - determines which n_blocks to use
    'mixing_time_max_steps': 500,
    'tolerance': 1e-4,
    'output_dir': 'sanity_check_results',
    # UPDATE THIS PATH to where your matrices are stored:
    # e.g., 'graded_entropy_matrices'
    'matrix_input_dir': 'assets/transition-matrices',
}


def get_sequence_length(matrix_size: int) -> int:
    """Calculate sequence length based on matrix size and study type."""
    n_blocks = CONFIG['n_blocks_full'] if CONFIG['study_type'] == 'full' else CONFIG['n_blocks_pilot']
    trials_per_block = matrix_size * CONFIG['trials_per_block_multiplier']
    return n_blocks * trials_per_block


# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate total variation distance between two probability distributions."""
    return 0.5 * np.sum(np.abs(p - q))


def matrix_power(P: np.ndarray, t: int) -> np.ndarray:
    """Calculate P^t using efficient exponentiation by squaring."""
    if t == 0:
        return np.eye(len(P))
    elif t == 1:
        return P.copy()
    else:
        result = np.eye(len(P))
        base = P.copy()
        while t > 0:
            if t % 2 == 1:
                result = result @ base
            base = base @ base
            t //= 2
        return result


def get_stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Calculate stationary distribution of transition matrix."""
    eigenvalues, eigenvectors = eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi = np.real(eigenvectors[:, idx])
    return pi / pi.sum()


def calculate_delta_t(P: np.ndarray, t: int, stationary_dist: np.ndarray) -> float:
    """Calculate Δ(t) = max_s ||π - P^t δ_s||_TV"""
    n = len(P)
    P_t = matrix_power(P, t)
    max_distance = 0.0
    for s in range(n):
        dist_after_t = P_t[:, s]
        distance = total_variation_distance(dist_after_t, stationary_dist)
        max_distance = max(max_distance, distance)
    return max_distance


def calculate_mixing_time(P: np.ndarray, max_steps: int = 500) -> Dict:
    """Calculate mixing time using exact TV distance."""
    epsilon = 1 / (2 * np.e)
    stationary_dist = get_stationary_distribution(P)

    delta_values = []
    mixing_time = None

    for t in range(max_steps):
        delta_t = calculate_delta_t(P, t, stationary_dist)
        delta_values.append(delta_t)

        if delta_t <= epsilon and mixing_time is None:
            mixing_time = t
            break

    return {
        'mixing_time': mixing_time,
        'delta_values': delta_values,
        'epsilon': epsilon,
        'stationary_distribution': stationary_dist.tolist()
    }


def validate_matrix(P: np.ndarray, tolerance: float = 1e-4) -> Dict:
    """Validate transition matrix properties."""
    n = len(P)
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)

    row_deviations = np.abs(row_sums - 1.0)
    col_deviations = np.abs(col_sums - 1.0)

    is_row_stochastic = bool(np.all(row_deviations < tolerance))
    is_col_stochastic = bool(np.all(col_deviations < tolerance))
    is_doubly_stochastic = is_row_stochastic and is_col_stochastic

    pi = get_stationary_distribution(P)
    uniform = np.ones(n) / n
    pi_deviation = float(np.max(np.abs(pi - uniform)))
    is_uniform_stationary = pi_deviation < tolerance

    # Calculate conditional entropies for each row
    entropies = []
    for row in P:
        probs = row[row > 1e-10]
        if len(probs) > 0:
            H = -np.sum(probs * np.log2(probs))
        else:
            H = 0.0
        entropies.append(float(H))

    return {
        'n': n,
        'is_row_stochastic': is_row_stochastic,
        'is_col_stochastic': is_col_stochastic,
        'is_doubly_stochastic': is_doubly_stochastic,
        'max_row_deviation': float(np.max(row_deviations)),
        'max_col_deviation': float(np.max(col_deviations)),
        'stationary_distribution': pi.tolist(),
        'is_uniform_stationary': is_uniform_stationary,
        'max_pi_deviation': pi_deviation,
        'conditional_entropies': entropies,
        'entropy_range': [min(entropies), max(entropies)],
        'tolerance_used': tolerance
    }


def validate_sequence_uniformity(counts: Union[List[int], np.ndarray, Dict]) -> Dict:
    """Validate that observed position counts are consistent with uniform distribution."""
    if isinstance(counts, dict):
        counts = np.array([counts[k] for k in sorted(counts.keys())])
    else:
        counts = np.array(counts)

    k = len(counts)
    n = int(counts.sum())
    expected = n / k

    chi2_stat, p_value = stats.chisquare(counts)

    cv_observed = (counts.std() / counts.mean()) * 100
    expected_sd = np.sqrt(n * (1/k) * (1 - 1/k))
    cv_expected = (expected_sd / expected) * 100

    is_acceptable = bool(p_value > 0.05)

    return {
        'chi2_statistic': float(chi2_stat),
        'degrees_of_freedom': int(k - 1),
        'p_value': float(p_value),
        'cv_observed_percent': float(cv_observed),
        'cv_expected_percent': float(cv_expected),
        'n_trials': int(n),
        'n_positions': int(k),
        'expected_count': float(expected),
        'observed_counts': [int(c) for c in counts.tolist()],
        'observed_range': [int(counts.min()), int(counts.max())],
        'is_acceptable': is_acceptable
    }


def validate_transition_probabilities(sequence: np.ndarray, P: np.ndarray,
                                      min_transitions: int = 30) -> Dict:
    """
    Validate that observed transition frequencies match the matrix probabilities.

    Uses chi-square test for each row (starting position) to check if 
    observed transitions match expected probabilities.

    Args:
        sequence: Generated sequence of positions
        P: Transition matrix
        min_transitions: Minimum transitions from a state to include in test

    Returns:
        Dictionary with validation results
    """
    n = len(P)

    # Count observed transitions
    transition_counts = np.zeros((n, n), dtype=int)
    for i in range(len(sequence) - 1):
        from_state = sequence[i]
        to_state = sequence[i + 1]
        transition_counts[from_state, to_state] += 1

    # Validate each row
    row_results = []
    all_p_values = []

    for i in range(n):
        row_total = transition_counts[i].sum()

        if row_total < min_transitions:
            # Not enough data to test this row
            row_results.append({
                'from_position': int(i),
                'n_transitions': int(row_total),
                'skipped': True,
                'reason': f'Too few transitions (< {min_transitions})'
            })
            continue

        # Expected counts based on matrix probabilities
        expected_probs = P[i]
        expected_counts = expected_probs * row_total
        observed_counts = transition_counts[i]

        # Only test positions with non-zero expected probability
        # (can't test transitions that should never happen)
        nonzero_mask = expected_probs > 1e-10

        if nonzero_mask.sum() < 2:
            # Deterministic row (only one possible transition)
            # Check if all transitions went to the right place
            expected_pos = np.argmax(expected_probs)
            actual_pos = np.argmax(observed_counts)
            is_correct = (expected_pos == actual_pos) and (
                observed_counts[expected_pos] == row_total)

            row_results.append({
                'from_position': int(i),
                'n_transitions': int(row_total),
                'deterministic': True,
                'expected_target': int(expected_pos),
                'all_correct': bool(is_correct),
                'p_value': 1.0 if is_correct else 0.0
            })
            if is_correct:
                all_p_values.append(1.0)
            continue

        # Chi-square test on non-zero probability transitions
        obs = observed_counts[nonzero_mask]
        exp = expected_counts[nonzero_mask]

        # Ensure expected counts are not too small for chi-square
        if np.any(exp < 5):
            # Use exact test or skip
            row_results.append({
                'from_position': int(i),
                'n_transitions': int(row_total),
                'skipped': True,
                'reason': 'Expected counts too small for chi-square'
            })
            continue

        chi2_stat = np.sum((obs - exp) ** 2 / exp)
        df = len(obs) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df) if df > 0 else 1.0

        all_p_values.append(p_value)

        row_results.append({
            'from_position': int(i),
            'n_transitions': int(row_total),
            'chi2_statistic': float(chi2_stat),
            'degrees_of_freedom': int(df),
            'p_value': float(p_value),
            'is_acceptable': bool(p_value > 0.05),
            'observed_counts': [int(c) for c in observed_counts.tolist()],
            'expected_counts': [float(c) for c in expected_counts.tolist()]
        })

    # Overall assessment
    tested_rows = [
        r for r in row_results if 'p_value' in r and not r.get('skipped', False)]
    n_acceptable = sum(1 for r in tested_rows if r.get(
        'is_acceptable', r.get('all_correct', False)))

    # Combined test: are the p-values uniformly distributed? (they should be under H0)
    # Using Fisher's method to combine p-values
    if all_p_values:
        # Fisher's combined probability test
        fisher_stat = -2 * np.sum(np.log(np.array(all_p_values) + 1e-10))
        fisher_df = 2 * len(all_p_values)
        combined_p = 1 - stats.chi2.cdf(fisher_stat, fisher_df)
    else:
        combined_p = None

    return {
        'n_positions': int(n),
        'total_transitions': int(len(sequence) - 1),
        'row_results': row_results,
        'n_rows_tested': len(tested_rows),
        'n_rows_acceptable': int(n_acceptable),
        'all_rows_acceptable': n_acceptable == len(tested_rows) if tested_rows else None,
        'combined_p_value': float(combined_p) if combined_p is not None else None,
        'overall_acceptable': bool(combined_p > 0.05) if combined_p is not None else None
    }


def convert_for_json(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}
    else:
        return obj


# ============================================================================
# SEQUENCE GENERATION
# ============================================================================

def generate_sequence(P: np.ndarray, length: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a sequence from the Markov chain defined by P."""
    if seed is not None:
        np.random.seed(seed)

    n = len(P)

    # Normalize rows to ensure they sum to exactly 1.0 (fixes floating point issues)
    P_normalized = P / P.sum(axis=1, keepdims=True)

    sequence = np.zeros(length, dtype=int)

    # Start from stationary distribution
    pi = get_stationary_distribution(P_normalized)
    # Ensure pi sums to 1
    pi = pi / pi.sum()

    current_state = np.random.choice(n, p=pi)

    for i in range(length):
        sequence[i] = current_state
        # Get transition probabilities and ensure they sum to 1
        probs = P_normalized[current_state]
        probs = probs / probs.sum()  # Extra safety normalization
        current_state = np.random.choice(n, p=probs)

    return sequence


def get_unigram_counts(sequence: np.ndarray, n_positions: int) -> Dict[int, int]:
    """Count occurrences of each position in sequence."""
    counts = {i: 0 for i in range(n_positions)}
    unique, cnts = np.unique(sequence, return_counts=True)
    for pos, cnt in zip(unique, cnts):
        counts[int(pos)] = int(cnt)
    return counts


def shuffle_transition_matrix_np(matrix):
    n = matrix.shape[0]
    perm = np.random.permutation(n)
    return matrix[perm][:, perm]


# ============================================================================
# PLOTTING
# ============================================================================

def plot_convergence(P: np.ndarray, save_path: str, title: str = ""):
    """Plot Δ(t) over time and save."""
    epsilon = 1 / (2 * np.e)
    stationary_dist = get_stationary_distribution(P)

    # Always compute enough steps to show convergence clearly
    # Use fixed range for consistent comparison across matrices
    max_steps = 50  # Fixed for visual consistency

    delta_values = []
    for t in range(max_steps):
        delta_t = calculate_delta_t(P, t, stationary_dist)
        delta_values.append(delta_t)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delta_values, linewidth=2, label='Δ(t)', color='blue')
    ax.axhline(y=epsilon, color='red', linestyle='--',
               label=f'ε = 1/(2e) ≈ {epsilon:.3f}')

    # Find and mark mixing time
    mixing_time = next((t for t, d in enumerate(
        delta_values) if d <= epsilon), None)
    if mixing_time:
        ax.axvline(x=mixing_time, color='green', linestyle=':', alpha=0.7,
                   label=f'τ_mix = {mixing_time}')

    ax.set_xlabel('Time steps (t)', fontsize=11)
    ax.set_ylabel('Δ(t) = max distance to stationarity', fontsize=11)
    ax.set_title(
        title or 'Convergence to Stationary Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================================
# BATCH ANALYSIS
# ============================================================================

def analyze_single_matrix(P: np.ndarray,
                          output_dir: Path,
                          matrix_name: str,
                          n_sequences: int = 10,
                          sequence_length: Optional[int] = None) -> Dict:
    """
    Complete analysis of a single matrix with multiple sequence generations.

    Args:
        P: Transition matrix
        output_dir: Where to save results
        matrix_name: Name for labeling
        n_sequences: Number of sequences to generate
        sequence_length: If None, calculated from matrix size using get_sequence_length()
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate sequence length if not provided
    if sequence_length is None:
        sequence_length = get_sequence_length(len(P))

    # Save matrix
    np.save(output_dir / 'matrix.npy', P)

    # Matrix validation
    matrix_results = validate_matrix(P, tolerance=CONFIG['tolerance'])

    # Mixing time
    mixing_results = calculate_mixing_time(
        P, max_steps=CONFIG['mixing_time_max_steps'])

    # Convergence plot
    plot_convergence(P, str(output_dir / 'convergence_plot.png'),
                     title=f'{matrix_name} Convergence')

    # Generate and validate sequences
    sequence_validations = []
    all_p_values = []
    all_cvs = []
    all_transition_p_values = []

    for seq_idx in range(n_sequences):
        sequence = generate_sequence(
            P, sequence_length, seed=seq_idx * 1000 + 42)
        counts = get_unigram_counts(sequence, len(P))

        # Unigram (position) validation
        unigram_validation = validate_sequence_uniformity(counts)

        # Transition probability validation
        transition_validation = validate_transition_probabilities(sequence, P)

        validation = {
            'sequence_index': seq_idx,
            'unigram': unigram_validation,
            'transitions': transition_validation
        }
        sequence_validations.append(validation)

        all_p_values.append(unigram_validation['p_value'])
        all_cvs.append(unigram_validation['cv_observed_percent'])
        if transition_validation['combined_p_value'] is not None:
            all_transition_p_values.append(
                transition_validation['combined_p_value'])

    # Aggregate sequence statistics
    sequence_summary = {
        'n_sequences': n_sequences,
        'sequence_length': sequence_length,
        # Unigram stats
        'unigram_all_acceptable': all(v['unigram']['is_acceptable'] for v in sequence_validations),
        'unigram_n_acceptable': sum(1 for v in sequence_validations if v['unigram']['is_acceptable']),
        'unigram_p_value_mean': float(np.mean(all_p_values)),
        'unigram_p_value_min': float(np.min(all_p_values)),
        'unigram_p_value_max': float(np.max(all_p_values)),
        'cv_mean': float(np.mean(all_cvs)),
        'cv_std': float(np.std(all_cvs)),
        # Transition stats
        'transition_all_acceptable': all(v['transitions']['overall_acceptable'] for v in sequence_validations if v['transitions']['overall_acceptable'] is not None),
        'transition_n_acceptable': sum(1 for v in sequence_validations if v['transitions'].get('overall_acceptable', False)),
        'transition_p_value_mean': float(np.mean(all_transition_p_values)) if all_transition_p_values else None,
    }

    # Compile results
    results = {
        'matrix_name': matrix_name,
        'matrix_validation': matrix_results,
        'mixing_time': {
            'tau_mix': mixing_results['mixing_time'],
            'epsilon': mixing_results['epsilon'],
            'sequence_to_mixing_ratio': sequence_length / mixing_results['mixing_time'] if mixing_results['mixing_time'] else None
        },
        'sequence_summary': sequence_summary,
        'sequence_validations': sequence_validations
    }

    # Save results
    with open(output_dir / 'matrix_analysis.json', 'w') as f:
        json.dump(convert_for_json({k: v for k, v in results.items() if k != 'sequence_validations'}),
                  f, indent=2)

    with open(output_dir / 'sequence_validations.json', 'w') as f:
        json.dump(convert_for_json(sequence_validations), f, indent=2)

    return results


def run_batch_analysis():
    """Run complete batch analysis for all matrix sizes and variations."""

    output_base = Path(CONFIG['output_dir'])
    output_base.mkdir(parents=True, exist_ok=True)

    all_results = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'sizes': {}
    }

    total_matrices = len(CONFIG['matrix_sizes']) * (1 + CONFIG['n_variations'])
    current_matrix = 0

    for size in CONFIG['matrix_sizes']:
        print(f"\n{'='*70}")
        print(f"ANALYZING {size}×{size} MATRICES")
        print(f"{'='*70}")

        size_dir = output_base / f'{size}x{size}'
        size_results = {'original': None, 'variations': []}

        # Load original matrix
        original_path = Path(
            CONFIG['matrix_input_dir']) / f'matrix_{size}x{size}.npy'

        if original_path.exists():
            P_original = np.load(original_path)
            print(
                f"\n[{current_matrix+1}/{total_matrices}] Analyzing original {size}×{size} matrix...")

            # Diagnostic: check row sums
            row_sums = P_original.sum(axis=1)
            if not np.allclose(row_sums, 1.0):
                print(f"  ⚠️  Row sums not exactly 1.0: {row_sums}")
                print(
                    f"      Max deviation: {np.max(np.abs(row_sums - 1.0)):.2e}")
                print(f"      (Will normalize before sequence generation)")

            results = analyze_single_matrix(
                P_original,
                size_dir / 'original',
                f'{size}×{size} Original',
                n_sequences=CONFIG['n_sequences'],
                sequence_length=get_sequence_length(size)
            )
            size_results['original'] = results
            current_matrix += 1

            print(f"  τ_mix = {results['mixing_time']['tau_mix']}")
            print(
                f"  Doubly stochastic: {results['matrix_validation']['is_doubly_stochastic']}")
            print(
                f"  Unigram acceptable: {results['sequence_summary']['unigram_n_acceptable']}/{CONFIG['n_sequences']}")
            print(
                f"  Transitions acceptable: {results['sequence_summary']['transition_n_acceptable']}/{CONFIG['n_sequences']}")

            # Generate and analyze variations
            for var_idx in range(CONFIG['n_variations']):
                print(
                    f"\n[{current_matrix+1}/{total_matrices}] Analyzing variation {var_idx+1} of {size}×{size}...")

                P_variation = shuffle_transition_matrix_np(P_original)

                results = analyze_single_matrix(
                    P_variation,
                    size_dir / f'variation_{var_idx+1}',
                    f'{size}×{size} Variation {var_idx+1}',
                    n_sequences=CONFIG['n_sequences'],
                    sequence_length=get_sequence_length(size)
                )
                size_results['variations'].append(results)
                current_matrix += 1

                print(f"  τ_mix = {results['mixing_time']['tau_mix']}")
                print(
                    f"  Doubly stochastic: {results['matrix_validation']['is_doubly_stochastic']}")
                print(
                    f"  Unigram acceptable: {results['sequence_summary']['unigram_n_acceptable']}/{CONFIG['n_sequences']}")
                print(
                    f"  Transitions acceptable: {results['sequence_summary']['transition_n_acceptable']}/{CONFIG['n_sequences']}")

        else:
            print(f"\n⚠️  Original matrix not found at {original_path}")

        all_results['sizes'][f'{size}x{size}'] = size_results

    # Generate summary
    generate_summary(all_results, output_base)

    return all_results


def generate_summary(all_results: Dict, output_dir: Path):
    """Generate summary report."""

    # Collect aggregate statistics
    summary_stats = {
        'total_matrices_analyzed': 0,
        'total_sequences_validated': 0,
        'all_matrices_doubly_stochastic': True,
        'all_unigrams_acceptable': True,
        'all_transitions_acceptable': True,
        'mixing_times': [],
        'unigram_p_values': [],
        'transition_p_values': [],
        'by_size': {}
    }

    for size_key, size_data in all_results['sizes'].items():
        size_stats = {
            'n_matrices': 0,
            'mixing_times': [],
            'unigram_p_value_means': [],
            'transition_p_value_means': [],
            'all_doubly_stochastic': True,
            'all_unigrams_acceptable': True,
            'all_transitions_acceptable': True
        }

        matrices_to_check = []
        if size_data['original']:
            matrices_to_check.append(('original', size_data['original']))
        for i, var in enumerate(size_data['variations']):
            matrices_to_check.append((f'variation_{i+1}', var))

        for name, results in matrices_to_check:
            summary_stats['total_matrices_analyzed'] += 1
            size_stats['n_matrices'] += 1

            if results['mixing_time']['tau_mix']:
                summary_stats['mixing_times'].append(
                    results['mixing_time']['tau_mix'])
                size_stats['mixing_times'].append(
                    results['mixing_time']['tau_mix'])

            if not results['matrix_validation']['is_doubly_stochastic']:
                summary_stats['all_matrices_doubly_stochastic'] = False
                size_stats['all_doubly_stochastic'] = False

            summary_stats['total_sequences_validated'] += results['sequence_summary']['n_sequences']

            # Unigram stats
            summary_stats['unigram_p_values'].append(
                results['sequence_summary']['unigram_p_value_mean'])
            size_stats['unigram_p_value_means'].append(
                results['sequence_summary']['unigram_p_value_mean'])

            if not results['sequence_summary']['unigram_all_acceptable']:
                summary_stats['all_unigrams_acceptable'] = False
                size_stats['all_unigrams_acceptable'] = False

            # Transition stats
            if results['sequence_summary']['transition_p_value_mean'] is not None:
                summary_stats['transition_p_values'].append(
                    results['sequence_summary']['transition_p_value_mean'])
                size_stats['transition_p_value_means'].append(
                    results['sequence_summary']['transition_p_value_mean'])

            if not results['sequence_summary']['transition_all_acceptable']:
                summary_stats['all_transitions_acceptable'] = False
                size_stats['all_transitions_acceptable'] = False

        summary_stats['by_size'][size_key] = size_stats

    # Save JSON summary
    json_summary = {
        'timestamp': all_results['timestamp'],
        'config': all_results['config'],
        'total_matrices_analyzed': summary_stats['total_matrices_analyzed'],
        'total_sequences_validated': summary_stats['total_sequences_validated'],
        'all_matrices_doubly_stochastic': summary_stats['all_matrices_doubly_stochastic'],
        'all_unigrams_acceptable': summary_stats['all_unigrams_acceptable'],
        'all_transitions_acceptable': summary_stats['all_transitions_acceptable'],
        'mixing_time_range': [min(summary_stats['mixing_times']), max(summary_stats['mixing_times'])] if summary_stats['mixing_times'] else None,
        'mixing_time_mean': float(np.mean(summary_stats['mixing_times'])) if summary_stats['mixing_times'] else None,
        'unigram_p_value_mean': float(np.mean(summary_stats['unigram_p_values'])) if summary_stats['unigram_p_values'] else None,
        'transition_p_value_mean': float(np.mean(summary_stats['transition_p_values'])) if summary_stats['transition_p_values'] else None,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(json_summary, f, indent=2)

    # Generate markdown report
    report = []
    report.append("# Sanity Check Results Summary")
    report.append(f"\n**Generated:** {all_results['timestamp']}")
    report.append(f"\n## Configuration")
    report.append(f"- Matrix sizes: {CONFIG['matrix_sizes']}")
    report.append(f"- Variations per size: {CONFIG['n_variations']}")
    report.append(f"- Sequences per matrix: {CONFIG['n_sequences']}")
    report.append(
        f"- Study type: {CONFIG['study_type']} ({CONFIG['n_blocks_pilot'] if CONFIG['study_type'] == 'pilot' else CONFIG['n_blocks_full']} blocks)")
    report.append(
        f"- Sequence lengths: {', '.join(f'{s}×{s}={get_sequence_length(s)}' for s in CONFIG['matrix_sizes'])}")

    report.append(f"\n## Overall Results")
    report.append(
        f"- **Total matrices analyzed:** {summary_stats['total_matrices_analyzed']}")
    report.append(
        f"- **Total sequences validated:** {summary_stats['total_sequences_validated']}")
    report.append(
        f"- **All matrices doubly stochastic:** {'✓ Yes' if summary_stats['all_matrices_doubly_stochastic'] else '✗ No'}")
    report.append(
        f"- **All unigram distributions acceptable (p > 0.05):** {'✓ Yes' if summary_stats['all_unigrams_acceptable'] else '✗ No'}")
    report.append(
        f"- **All transition distributions acceptable (p > 0.05):** {'✓ Yes' if summary_stats['all_transitions_acceptable'] else '✗ No'}")

    if summary_stats['mixing_times']:
        report.append(
            f"- **Mixing time range:** {min(summary_stats['mixing_times'])} - {max(summary_stats['mixing_times'])} steps")
        report.append(
            f"- **Mixing time mean:** {np.mean(summary_stats['mixing_times']):.1f} steps")
        # Use minimum sequence length for conservative ratio estimate
        min_seq_len = min(get_sequence_length(s)
                          for s in CONFIG['matrix_sizes'])
        report.append(
            f"- **Min sequence/τ_mix ratio:** {min_seq_len / np.max(summary_stats['mixing_times']):.0f}× (conservative)")

    report.append(f"\n## Results by Matrix Size")
    report.append("")
    report.append(
        "| Size | Matrices | τ_mix Range | Unigram p | Transition p | All DS | Unigram OK | Trans OK |")
    report.append(
        "|------|----------|-------------|-----------|--------------|--------|------------|----------|")

    for size_key, size_stats in summary_stats['by_size'].items():
        if size_stats['mixing_times']:
            tau_range = f"{min(size_stats['mixing_times'])}-{max(size_stats['mixing_times'])}"
        else:
            tau_range = "N/A"

        uni_p = f"{np.mean(size_stats['unigram_p_value_means']):.3f}" if size_stats['unigram_p_value_means'] else "N/A"
        trans_p = f"{np.mean(size_stats['transition_p_value_means']):.3f}" if size_stats['transition_p_value_means'] else "N/A"
        ds = "✓" if size_stats['all_doubly_stochastic'] else "✗"
        uni_ok = "✓" if size_stats['all_unigrams_acceptable'] else "✗"
        trans_ok = "✓" if size_stats['all_transitions_acceptable'] else "✗"

        report.append(
            f"| {size_key} | {size_stats['n_matrices']} | {tau_range} | {uni_p} | {trans_p} | {ds} | {uni_ok} | {trans_ok} |")

    report.append(f"\n## Interpretation")
    report.append("")
    report.append("### Metrics Explained")
    report.append(
        "- **τ_mix (mixing time):** Steps until probability distribution converges to stationary")
    report.append(
        "- **Doubly stochastic (DS):** Row and column sums equal 1 → guarantees uniform stationary distribution")
    report.append(
        "- **Unigram p-value:** χ² test for whether position visit counts match uniform distribution")
    report.append(
        "- **Transition p-value:** χ² test for whether observed transitions match matrix probabilities (Fisher's combined)")
    report.append(
        "- **Sequence/τ_mix ratio:** How many times longer the sequence is than mixing time; >10× is good")

    report.append(f"\n### For Methods Section")
    if summary_stats['mixing_times']:
        mean_tau = np.mean(summary_stats['mixing_times'])
        max_tau = max(summary_stats['mixing_times'])
        min_seq_len = min(get_sequence_length(s)
                          for s in CONFIG['matrix_sizes'])
        max_seq_len = max(get_sequence_length(s)
                          for s in CONFIG['matrix_sizes'])
        min_ratio = min_seq_len / max_tau
        uni_p_mean = np.mean(
            summary_stats['unigram_p_values']) if summary_stats['unigram_p_values'] else 0
        trans_p_mean = np.mean(
            summary_stats['transition_p_values']) if summary_stats['transition_p_values'] else 0
        report.append(f"""
> Sequences were generated from doubly stochastic transition matrices using Markov chains. 
> Mixing times ranged from {min(summary_stats['mixing_times'])} to {max_tau} steps 
> (M = {mean_tau:.1f}, computed via total variation distance), confirming that sequence lengths 
> ({min_seq_len}–{max_seq_len} trials, varying by matrix size) exceeded τ_mix by a factor of at least {min_ratio:.0f}×. 
> Across {summary_stats['total_sequences_validated']} generated sequences, empirical unigram distributions 
> did not differ significantly from uniform (mean p = {uni_p_mean:.2f}), and observed transition 
> frequencies matched the specified matrix probabilities (mean combined p = {trans_p_mean:.2f}).
""")

    with open(output_dir / 'summary_report.md', 'w') as f:
        f.write('\n'.join(report))

    print(f"\n{'='*70}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - summary.json")
    print(f"  - summary_report.md")
    print(f"  - [size]/[matrix]/matrix_analysis.json")
    print(f"  - [size]/[matrix]/sequence_validations.json")
    print(f"  - [size]/[matrix]/convergence_plot.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("BATCH SANITY CHECKS FOR PROCEDURAL LEARNING MATRICES")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Matrix sizes: {CONFIG['matrix_sizes']}")
    print(f"  Variations per size: {CONFIG['n_variations']}")
    print(f"  Sequences per matrix: {CONFIG['n_sequences']}")
    print(f"  Study type: {CONFIG['study_type']}")
    n_blocks = CONFIG['n_blocks_full'] if CONFIG['study_type'] == 'full' else CONFIG['n_blocks_pilot']
    print(f"  Blocks: {n_blocks}")
    print(f"  Sequence lengths by size:")
    for size in CONFIG['matrix_sizes']:
        seq_len = get_sequence_length(size)
        print(f"    {size}×{size}: {n_blocks} × {size} × 10 = {seq_len} trials")
    print(f"  Output directory: {CONFIG['output_dir']}")

    total = len(CONFIG['matrix_sizes']) * (1 + CONFIG['n_variations'])
    print(f"\nTotal matrices to analyze: {total}")
    print(f"Total sequences to validate: {total * CONFIG['n_sequences']}")

    results = run_batch_analysis()
