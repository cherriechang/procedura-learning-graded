#! /usr/bin/env python3
"""
Mixing Time and Sequence Validation Analysis for Transition Matrices

This module provides tools to:
1. Calculate mixing time (τ_mix) using exact TV distance (works for ANY chain)
2. Validate that matrices are (approximately) doubly stochastic
3. Validate generated sequences have expected unigram distributions

Key formulas:
  Δ(t) = max_s ||π - P^t δ_s||_TV
  τ_mix = min{t : Δ(t) ≤ 1/(2e)}
  
For sequence validation:
  χ² goodness-of-fit test
  Coefficient of Variation (CV)

Author: Claude (for Cherrie Chang)
Project: Procedural Learning & Grammar Acquisition
"""

import numpy as np
from scipy.linalg import eig
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import json


# ============================================================================
# MATRIX VALIDATION
# ============================================================================

def validate_matrix(P: np.ndarray,
                    tolerance: float = 1e-4,
                    verbose: bool = True) -> Dict:
    """
    Validate transition matrix properties.

    Args:
        P: Transition matrix (n × n numpy array)
        tolerance: How close to 1.0 sums must be (default: 1e-4)
        verbose: Whether to print details

    Returns:
        Dictionary with validation results
    """
    n = len(P)
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)

    row_deviations = np.abs(row_sums - 1.0)
    col_deviations = np.abs(col_sums - 1.0)

    is_row_stochastic = np.all(row_deviations < tolerance)
    is_col_stochastic = np.all(col_deviations < tolerance)
    is_doubly_stochastic = is_row_stochastic and is_col_stochastic

    # Calculate actual stationary distribution
    eigenvalues, eigenvectors = eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()

    # Check if stationary is approximately uniform
    uniform = np.ones(n) / n
    pi_deviation = np.max(np.abs(pi - uniform))
    is_uniform_stationary = pi_deviation < tolerance

    results = {
        'n': n,
        'is_row_stochastic': is_row_stochastic,
        'is_col_stochastic': is_col_stochastic,
        'is_doubly_stochastic': is_doubly_stochastic,
        'max_row_deviation': float(np.max(row_deviations)),
        'max_col_deviation': float(np.max(col_deviations)),
        'stationary_distribution': pi.tolist(),
        'is_uniform_stationary': is_uniform_stationary,
        'max_pi_deviation': float(pi_deviation),
        'tolerance_used': tolerance
    }

    if verbose:
        print("="*70)
        print("MATRIX VALIDATION")
        print("="*70)
        print(f"\nMatrix size: {n} × {n}")
        print(f"Tolerance: {tolerance}")
        print(
            f"\nRow stochastic: {is_row_stochastic} (max deviation: {np.max(row_deviations):.2e})")
        print(
            f"Column stochastic: {is_col_stochastic} (max deviation: {np.max(col_deviations):.2e})")
        print(f"Doubly stochastic: {is_doubly_stochastic}")
        print(
            f"\nStationary distribution uniform: {is_uniform_stationary} (max deviation: {pi_deviation:.2e})")

        if not is_uniform_stationary:
            print("\n⚠️  WARNING: Stationary distribution is NOT uniform!")
            print("   This will cause uneven position visitation.")
            print(f"   π = {pi}")

    return results


# ============================================================================
# MIXING TIME CALCULATION
# ============================================================================

def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate total variation distance between two probability distributions.

    TV(p, q) = (1/2) * Σ|p_i - q_i|
    """
    return 0.5 * np.sum(np.abs(p - q))


def matrix_power(P: np.ndarray, t: int) -> np.ndarray:
    """
    Calculate P^t using efficient exponentiation by squaring.
    """
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


def calculate_delta_t(P: np.ndarray,
                      t: int,
                      stationary_dist: Optional[np.ndarray] = None) -> float:
    """
    Calculate Δ(t) = max_s ||π - P^t δ_s||_TV

    This measures the worst-case distance from stationarity after t steps.
    Works for ANY Markov chain (reversible or not).
    """
    n = len(P)

    if stationary_dist is None:
        # Calculate actual stationary distribution
        eigenvalues, eigenvectors = eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary_dist = np.real(eigenvectors[:, idx])
        stationary_dist = stationary_dist / stationary_dist.sum()

    P_t = matrix_power(P, t)
    max_distance = 0.0

    for s in range(n):
        # Distribution after t steps starting from state s = s-th column of P^t
        dist_after_t = P_t[:, s]
        distance = total_variation_distance(dist_after_t, stationary_dist)
        max_distance = max(max_distance, distance)

    return max_distance


def calculate_mixing_time(P: np.ndarray,
                          epsilon: Optional[float] = None,
                          max_steps: int = 10000,
                          verbose: bool = True) -> Dict:
    """
    Calculate mixing time τ_mix = min{t : Δ(t) ≤ ε}.

    Uses EXACT TV distance calculation (works for non-reversible chains).

    Args:
        P: Transition matrix
        epsilon: Threshold (default: 1/(2e) ≈ 0.184)
        max_steps: Maximum steps to check
        verbose: Print progress

    Returns:
        Dictionary with mixing_time, delta_values, epsilon
    """
    if epsilon is None:
        epsilon = 1 / (2 * np.e)  # ≈ 0.184

    # Get stationary distribution once
    eigenvalues, eigenvectors = eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    stationary_dist = np.real(eigenvectors[:, idx])
    stationary_dist = stationary_dist / stationary_dist.sum()

    delta_values = []
    mixing_time = None

    for t in range(max_steps):
        delta_t = calculate_delta_t(P, t, stationary_dist)
        delta_values.append(delta_t)

        if verbose and t % 100 == 0:
            print(f"t={t:4d}: Δ(t)={delta_t:.6f}")

        if delta_t <= epsilon and mixing_time is None:
            mixing_time = t
            if verbose:
                print(f"\n✓ Mixing time reached at t={t}")
                print(f"  Δ({t}) = {delta_t:.6f} ≤ {epsilon:.6f}")
            break

    if mixing_time is None and verbose:
        print(f"\n⚠ Warning: Mixing time not reached within {max_steps} steps")
        print(f"  Final Δ({max_steps-1}) = {delta_values[-1]:.6f}")

    return {
        'mixing_time': mixing_time,
        'delta_values': delta_values,
        'epsilon': epsilon,
        'stationary_distribution': stationary_dist.tolist()
    }


def calculate_eigenvalue_mixing_time(P: np.ndarray, verbose: bool = True) -> Dict:
    """
    Calculate mixing time using eigenvalue method (APPROXIMATION).

    NOTE: This is only accurate for REVERSIBLE chains!
    For non-reversible chains, use calculate_mixing_time() instead.

    Formula: τ_mix ≈ 1 / (1 - |λ₂|)
    """
    eigenvalues, _ = eig(P)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]

    lambda_1 = eigenvalues[0]
    lambda_2 = eigenvalues[1]

    # Check if chain might be non-reversible (complex eigenvalues)
    has_complex = any(np.abs(np.imag(eigenvalues)) > 1e-10)

    gap = 1 - np.abs(lambda_2)
    tau_approx = 1 / gap if gap > 0 else np.inf

    if verbose:
        print(f"Second eigenvalue λ₂ = {lambda_2}")
        print(f"Spectral gap = {gap:.6f}")
        print(f"Approximate τ_mix = {tau_approx:.1f}")
        if has_complex:
            print("⚠️  Note: Chain has complex eigenvalues (non-reversible)")
            print("   Eigenvalue approximation may be inaccurate!")

    return {
        'eigenvalues': eigenvalues.tolist(),
        'lambda_2': complex(lambda_2),
        'spectral_gap': float(gap),
        'mixing_time_approx': float(tau_approx),
        'has_complex_eigenvalues': has_complex,
        'warning': 'Approximation only valid for reversible chains' if has_complex else None
    }


# ============================================================================
# SEQUENCE VALIDATION
# ============================================================================

def validate_sequence_uniformity(counts: Union[List[int], np.ndarray, Dict]) -> Dict:
    """
    Validate that observed position counts are consistent with uniform distribution.

    Args:
        counts: Array/list of observed counts per position, or dict {position: count}

    Returns:
        Dictionary with chi-square test results and CV
    """
    if isinstance(counts, dict):
        counts = np.array([counts[k] for k in sorted(counts.keys())])
    else:
        counts = np.array(counts)

    k = len(counts)
    n = int(counts.sum())
    expected = n / k

    # Chi-square goodness-of-fit test
    chi2_stat, p_value = stats.chisquare(counts)

    # Coefficient of variation
    cv_observed = (counts.std() / counts.mean()) * 100

    # Expected CV from multinomial sampling
    # Var(count_i) = n * (1/k) * (1 - 1/k)
    expected_sd = np.sqrt(n * (1/k) * (1 - 1/k))
    cv_expected = (expected_sd / expected) * 100

    # Determine if result is acceptable
    is_acceptable = p_value > 0.05

    return {
        'chi2_statistic': float(chi2_stat),
        'degrees_of_freedom': k - 1,
        'p_value': float(p_value),
        'cv_observed_percent': float(cv_observed),
        'cv_expected_percent': float(cv_expected),
        'n_trials': n,
        'n_positions': k,
        'expected_count': float(expected),
        'observed_counts': counts.tolist(),
        'observed_range': [int(counts.min()), int(counts.max())],
        'is_acceptable': is_acceptable,
        'interpretation': 'Consistent with uniform' if is_acceptable else 'Significant deviation from uniform'
    }


def format_validation_report(sequence_results: Dict,
                             mixing_results: Dict,
                             matrix_results: Dict) -> str:
    """
    Format a complete validation report for methods section.
    """
    report = []
    report.append("="*70)
    report.append("SEQUENCE VALIDATION REPORT")
    report.append("="*70)

    # Matrix properties
    report.append("\n## Matrix Properties")
    report.append(f"Size: {matrix_results['n']}×{matrix_results['n']}")
    report.append(
        f"Doubly stochastic: {matrix_results['is_doubly_stochastic']}")
    report.append(
        f"  Max row deviation: {matrix_results['max_row_deviation']:.2e}")
    report.append(
        f"  Max col deviation: {matrix_results['max_col_deviation']:.2e}")
    report.append(
        f"Uniform stationary: {matrix_results['is_uniform_stationary']}")
    report.append(
        f"  Max π deviation: {matrix_results['max_pi_deviation']:.2e}")

    # Mixing time
    report.append("\n## Mixing Time Analysis")
    report.append(f"τ_mix (exact TV): {mixing_results['mixing_time']} steps")
    report.append(f"Threshold ε: 1/(2e) ≈ {mixing_results['epsilon']:.4f}")

    # Sequence validation
    sr = sequence_results
    report.append("\n## Sequence Uniformity")
    report.append(f"Sequence length: {sr['n_trials']} trials")
    report.append(f"Positions: {sr['n_positions']}")
    report.append(
        f"Ratio (length/τ_mix): {sr['n_trials']/mixing_results['mixing_time']:.1f}×")
    report.append(
        f"\nχ²({sr['degrees_of_freedom']}) = {sr['chi2_statistic']:.2f}, p = {sr['p_value']:.3f}")
    report.append(f"CV observed: {sr['cv_observed_percent']:.1f}%")
    report.append(
        f"CV expected (multinomial): {sr['cv_expected_percent']:.1f}%")
    report.append(
        f"Count range: {sr['observed_range'][0]} - {sr['observed_range'][1]} (expected: {sr['expected_count']:.0f})")
    report.append(f"\nConclusion: {sr['interpretation']}")

    # Summary for methods section
    report.append("\n" + "="*70)
    report.append("SUMMARY")
    report.append("="*70)
    report.append(f"""
Sequences were generated from the transition matrix using a Markov chain. 
To verify adequate mixing, we computed the mixing time (τ_mix = {mixing_results['mixing_time']} steps 
via total variation distance), confirming that sequence length ({sr['n_trials']} trials) 
exceeded τ_mix by a factor of {sr['n_trials']/mixing_results['mixing_time']:.0f}×. 
Empirical unigram distributions did not differ significantly from uniform 
(χ²({sr['degrees_of_freedom']}) = {sr['chi2_statistic']:.2f}, p = {sr['p_value']:.2f}; CV = {sr['cv_observed_percent']:.1f}%).
""")

    return "\n".join(report)


# ============================================================================
# COMPLETE ANALYSIS FUNCTION
# ============================================================================

def analyze_matrix_complete(P: np.ndarray,
                            sequence_counts: Optional[Union[List,
                                                            Dict]] = None,
                            max_steps: int = 1000,
                            tolerance: float = 1e-4,
                            verbose: bool = True) -> Dict:
    """
    Complete analysis of a transition matrix and optionally a generated sequence.

    Args:
        P: Transition matrix
        sequence_counts: Optional observed unigram counts from generated sequence
        max_steps: Max steps for mixing time calculation
        tolerance: Tolerance for doubly stochastic check
        verbose: Print detailed output

    Returns:
        Dictionary with all analysis results
    """
    results = {}

    # 1. Validate matrix
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: MATRIX VALIDATION")
        print("="*70)
    results['matrix'] = validate_matrix(
        P, tolerance=tolerance, verbose=verbose)

    # 2. Calculate mixing time (exact method - works for non-reversible)
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: MIXING TIME (Exact TV Distance)")
        print("="*70)
    results['mixing_time'] = calculate_mixing_time(
        P, max_steps=max_steps, verbose=verbose)

    # 3. Eigenvalue analysis (for reference, with warning about reversibility)
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: EIGENVALUE ANALYSIS (Reference Only)")
        print("="*70)
    results['eigenvalues'] = calculate_eigenvalue_mixing_time(
        P, verbose=verbose)

    # 4. Sequence validation (if counts provided)
    if sequence_counts is not None:
        if verbose:
            print("\n" + "="*70)
            print("STEP 4: SEQUENCE VALIDATION")
            print("="*70)
        results['sequence'] = validate_sequence_uniformity(sequence_counts)

        if verbose:
            sr = results['sequence']
            print(f"\nSequence length: {sr['n_trials']} trials")
            print(f"Positions: {sr['n_positions']}")
            print(f"Expected count per position: {sr['expected_count']:.1f}")
            print(
                f"Observed range: {sr['observed_range'][0]} - {sr['observed_range'][1]}")
            print(
                f"\nχ²({sr['degrees_of_freedom']}) = {sr['chi2_statistic']:.2f}, p = {sr['p_value']:.3f}")
            print(f"CV observed: {sr['cv_observed_percent']:.1f}%")
            print(f"CV expected: {sr['cv_expected_percent']:.1f}%")
            print(f"\nResult: {sr['interpretation']}")

        # Generate formatted report
        results['report'] = format_validation_report(
            results['sequence'],
            results['mixing_time'],
            results['matrix']
        )

        if verbose:
            print(results['report'])

    return results


def plot_convergence(P: np.ndarray,
                     max_steps: int = 500,
                     save_path: Optional[str] = None) -> List[float]:
    """
    Plot Δ(t) over time to visualize convergence.
    """
    epsilon = 1 / (2 * np.e)

    # Get stationary distribution
    eigenvalues, eigenvectors = eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    stationary_dist = np.real(eigenvectors[:, idx])
    stationary_dist = stationary_dist / stationary_dist.sum()

    delta_values = []
    for t in range(max_steps):
        delta_t = calculate_delta_t(P, t, stationary_dist)
        delta_values.append(delta_t)
        if t % 100 == 0:
            print(f"Computing t={t}...")

    plt.figure(figsize=(10, 6))
    plt.plot(delta_values, linewidth=2, label='Δ(t)')
    plt.axhline(y=epsilon, color='r', linestyle='--',
                label=f'ε = 1/(2e) ≈ {epsilon:.3f}')

    # Find and mark mixing time
    mixing_time = next((t for t, d in enumerate(
        delta_values) if d <= epsilon), None)
    if mixing_time:
        plt.axvline(x=mixing_time, color='g', linestyle=':', alpha=0.7,
                    label=f'τ_mix = {mixing_time}')

    plt.xlabel('Time steps (t)', fontsize=12)
    plt.ylabel('Δ(t) = max distance to stationarity', fontsize=12)
    plt.title('Convergence to Stationary Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✓ Plot saved to {save_path}")

    plt.show()
    return delta_values


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Matrix to analyze
    # Using 8x8 matrix as example
    P_8x8_original = np.load("assets/transition-matrices/matrix_8x8.npy")

    # Example observed counts from pilot (560 trials)
    pilot_counts = {0: 69, 1: 59, 2: 73, 3: 61, 4: 73, 5: 75, 6: 73, 7: 77}

    # Run complete analysis
    results = analyze_matrix_complete(
        P_8x8_original,
        sequence_counts=pilot_counts,
        max_steps=500,
        tolerance=1e-4,
        verbose=True
    )

    # Save results
    # Convert numpy types to native Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        else:
            return obj

    results_serializable = convert_for_json({
        'matrix': results['matrix'],
        'mixing_time': {
            'mixing_time': results['mixing_time']['mixing_time'],
            'epsilon': results['mixing_time']['epsilon']
        },
        'sequence': results.get('sequence')
    })

    with open('matrix_validation_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("\n✓ Results saved to matrix_validation_results.json")
