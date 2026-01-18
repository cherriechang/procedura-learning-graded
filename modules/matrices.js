// Each matrix is sorted by conditional entropy (low to high)

const TRANSITION_MATRICES = {
	4: {
		matrix: [
			[0.02737, 0.97263, 0.0, 0.0],
			[0.0, 0.0, 0.701325, 0.298675],
			[0.585862, 0.0, 0.060265, 0.353873],
			[0.386768, 0.027368, 0.238412, 0.347452],
		],
		entropies: [0.181, 0.88, 1.227, 1.695],
		size: 4,
	},

	5: {
		matrix: [
			[0.017413, 0.982584, 0.0, 0.0, 0.0],
			[0.203084, 0.0, 0.796916, 0.0, 0.0],
			[0.033786, 0.0, 0.0, 0.483094, 0.48312],
			[0.459284, 0.0, 0.026421, 0.25716, 0.257135],
			[0.286432, 0.017416, 0.176664, 0.259744, 0.259744],
		],
		entropies: [0.127, 0.728, 1.179, 1.662, 2.071],
		size: 5,
	},

	6: {
		matrix: [
			[0.0, 0.988218, 0.0, 0.0, 0.0, 0.011782],
			[0.0, 0.0, 0.145857, 0.854143, 0.0, 0.0],
			[0.497693, 0.0, 0.004736, 0.0, 0.497571, 0.0],
			[0.065516, 0.0, 0.369215, 0.0, 0.065542, 0.499727],
			[0.227006, 0.0, 0.267032, 0.004171, 0.227057, 0.274734],
			[0.209785, 0.011782, 0.213159, 0.141683, 0.209827, 0.213764],
		],
		entropies: [0.092, 0.599, 1.039, 1.546, 2.025, 2.371],
		size: 6,
	},

	7: {
		matrix: [
			[0.0, 0.991152, 0.0, 0.0, 0.0, 0.008848, 0.0],
			[0.0, 0.0, 0.0, 0.119138, 0.880862, 0.0, 0.0],
			[0.64294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35706],
			[0.0, 0.0, 0.455141, 0.072325, 0.0, 0.456333, 0.016201],
			[0.0, 0.0, 0.16607, 0.429808, 0.0, 0.156088, 0.248034],
			[0.183307, 0.0, 0.204199, 0.204164, 0.0, 0.204177, 0.204153],
			[0.173758, 0.008848, 0.174586, 0.17456, 0.119121, 0.174574, 0.174553],
		],
		entropies: [0.073, 0.527, 0.94, 1.404, 1.871, 2.321, 2.623],
		size: 7,
	},

	8: {
		matrix: [
			[0.0, 0.992978, 0.0, 0.0, 0.007022, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.897971, 0.0, 0.102029, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.27616, 0.00076, 0.72308, 0.0, 0.0],
			[0.647184, 0.0, 0.0, 0.0, 0.197105, 0.0, 0.0, 0.155754],
			[0.004297, 0.0, 0.0, 0.151846, 0.071202, 0.0, 0.516503, 0.256152],
			[0.01712, 0.0, 0.0, 0.238284, 0.278302, 0.012355, 0.180366, 0.273573],
			[0.182971, 0.0, 0.0, 0.185204, 0.195132, 0.116054, 0.154599, 0.16604],
			[0.148465, 0.007022, 0.102023, 0.148507, 0.148481, 0.148455, 0.148526, 0.148521],
		],
		entropies: [0.06, 0.475, 0.859, 1.286, 1.714, 2.143, 2.566, 2.838],
		size: 8,
	},
};

/**
 * Shuffles a transition matrix by permuting states while preserving the probability distribution.
 * This reorders which position corresponds to which entropy level without changing the underlying
 * transition structure.
 *
 * @param {number[][]} matrix - Square transition matrix
 * @returns {number[][]} Shuffled transition matrix with same structure but reordered states
 */
function shuffleTransitionMatrix(matrix) {
	const n = matrix.length;

	// Generate random permutation of indices [0, 1, 2, ..., n-1]
	const permutation = Array.from({length: n}, (_, i) => i);
	for (let i = n - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[permutation[i], permutation[j]] = [permutation[j], permutation[i]];
	}

	// Create new matrix by permuting both rows and columns
	const shuffled = Array(n).fill(null).map(() => Array(n).fill(0));

	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) {
			// Map position i to permutation[i] and position j to permutation[j]
			shuffled[i][j] = matrix[permutation[i]][permutation[j]];
		}
	}

	return shuffled;
}
// Export for use in experiment
export {TRANSITION_MATRICES, shuffleTransitionMatrix};
