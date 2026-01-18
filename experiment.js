import {TRANSITION_MATRICES, shuffleTransitionMatrix} from "./modules/matrices.js";

const CONFIG = {
	matrix_size: 4, // Will be randomly assigned: 4, 5, 6, 7, or 8
	n_blocks: 5,
	trials_per_block: 80,
	practice_trials: 8, // 2x matrix size for practice
	rsi: 120, // ms
	error_feedback_duration: 200,
	error_tone_duration: 100,
	correct_feedback_duration: 200,
	block_break_duration: 15000,
};

// Matrices are pre-sorted by entropy (Position 0 = lowest, Position N-1 = highest)
const MATRICES = {
	4: TRANSITION_MATRICES[4].matrix,
	5: TRANSITION_MATRICES[5].matrix,
	6: TRANSITION_MATRICES[6].matrix,
	7: TRANSITION_MATRICES[7].matrix,
	8: TRANSITION_MATRICES[8].matrix,
};

// Key mappings (position index -> keyboard key)
const KEY_MAPPINGS_4 = ["d", "f", "j", "k"];
const KEY_MAPPINGS_5 = ["s", "d", "f", "j", "k"];
const KEY_MAPPINGS_6 = ["s", "d", "f", "j", "k", "l"];
const KEY_MAPPINGS_7 = ["a", "s", "d", "f", "j", "k", "l"];
const KEY_MAPPINGS_8 = ["a", "s", "d", "f", "j", "k", "l", ";"];

const KEY_MAPPINGS = {
	4: KEY_MAPPINGS_4,
	5: KEY_MAPPINGS_5,
	6: KEY_MAPPINGS_6,
	7: KEY_MAPPINGS_7,
	8: KEY_MAPPINGS_8,
};

let experimentState = {
	participantId: "P" + Date.now(),
	matrixSize: null,
	transitionMatrix: null,
	keyMapping: null,
	sequence: [],
	currentBlock: 0,
	currentTrial: 0,
	trialData: [],
	blockData: [],
	startTime: null,
};

function randomChoice(array, probabilities) {
	const random = Math.random();
	let cumsum = 0;
	for (let i = 0; i < array.length; i++) {
		cumsum += probabilities[i];
		if (random < cumsum) {
			return array[i];
		}
	}
	return array[array.length - 1];
}

function generateSequence(matrix, nTrials) {
	const sequence = [];
	const nStates = matrix.length;

	// Start from random position
	let currentPos = Math.floor(Math.random() * nStates);
	sequence.push(currentPos);

	// Generate remaining trials
	for (let i = 1; i < nTrials; i++) {
		const transitionProbs = matrix[currentPos];
		const positions = Array.from({length: nStates}, (_, i) => i);
		currentPos = randomChoice(positions, transitionProbs);
		sequence.push(currentPos);
	}

	return sequence;
}

function calculateConditionalEntropy(transitionProbs) {
	let entropy = 0;
	for (let p of transitionProbs) {
		if (p > 0) {
			entropy -= p * Math.log2(p);
		}
	}
	return entropy;
}

function getPositionEntropies(matrix) {
	return matrix.map((row) => calculateConditionalEntropy(row));
}

function createStimulusDisplay(
	position = null,
	matrixSize,
	showKeys = false,
	feedbackMessage = "",
) {
	let html = '<div class="feedback-message">';
	if (feedbackMessage) {
		html += feedbackMessage;
	}
	html += "</div>";

	html += '<div class="stimulus-container">';
	for (let i = 0; i < matrixSize; i++) {
		const active = i === position ? "active" : "";
		html += `<div class="position-wrapper">`;
		if (showKeys) {
			html += `<div class="key-label">${KEY_MAPPINGS[matrixSize][i].toUpperCase()}</div>`;
		}
		html += `<div class="position-box ${active}" data-position="${i}"></div>`;
		html += "</div>";
	}
	html += "</div>";

	return html;
}

// Add keypress visual feedback
function setupKeyPressHandlers(matrixSize) {
	const keyMapping = KEY_MAPPINGS[matrixSize];

	// Remove any existing handlers to avoid duplicates
	document.removeEventListener("keydown", window.keyDownHandler);
	document.removeEventListener("keyup", window.keyUpHandler);

	// Create new handlers
	window.keyDownHandler = function (e) {
		const keyIndex = keyMapping.indexOf(e.key.toLowerCase());
		if (keyIndex !== -1) {
			const boxes = document.querySelectorAll(".position-box");
			if (boxes[keyIndex]) {
				boxes[keyIndex].classList.add("pressed");
			}
		}
	};

	window.keyUpHandler = function (e) {
		const keyIndex = keyMapping.indexOf(e.key.toLowerCase());
		if (keyIndex !== -1) {
			const boxes = document.querySelectorAll(".position-box");
			if (boxes[keyIndex]) {
				boxes[keyIndex].classList.remove("pressed");
			}
		}
	};

	// Attach the handlers
	document.addEventListener("keydown", window.keyDownHandler);
	document.addEventListener("keyup", window.keyUpHandler);
}

function playErrorTone() {
	const audioContext = new (window.AudioContext || window.webkitAudioContext)();
	const oscillator = audioContext.createOscillator();
	const gainNode = audioContext.createGain();

	oscillator.connect(gainNode);
	gainNode.connect(audioContext.destination);

	oscillator.frequency.value = 200; // Low frequency for error
	oscillator.type = "sine";

	gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);
	gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);

	oscillator.start(audioContext.currentTime);
	oscillator.stop(audioContext.currentTime + 0.1);
}

function initializeExperiment() {
	// Randomly assign matrix size
	const matrixSizes = [4, 5, 6, 7, 8];
	experimentState.matrixSize = matrixSizes[Math.floor(Math.random() * matrixSizes.length)];

	// Load transition matrix and shuffle it
	const originalMatrix = MATRICES[experimentState.matrixSize];
	experimentState.transitionMatrix = shuffleTransitionMatrix(originalMatrix);
	experimentState.keyMapping = KEY_MAPPINGS[experimentState.matrixSize];

	// Generate full sequence for all blocks
	const totalTrials = CONFIG.n_blocks * CONFIG.trials_per_block;
	experimentState.sequence = generateSequence(experimentState.transitionMatrix, totalTrials);

	// Calculate position entropies for data recording
	experimentState.positionEntropies = getPositionEntropies(experimentState.transitionMatrix);

	experimentState.startTime = Date.now();

	console.log("Experiment initialized:", {
		matrixSize: experimentState.matrixSize,
		totalTrials: totalTrials,
		entropies: experimentState.positionEntropies,
	});
}

const jsPsych = initJsPsych({
	on_finish: function () {
		// Add matrix information to all data rows
		jsPsych.data.addProperties({
			matrix_size: experimentState.matrixSize,
			transition_matrix: JSON.stringify(experimentState.transitionMatrix),
		});
		jsPsych.data.displayData("csv");
	},
});

let timeline = [];

const welcome = {
	type: jsPsychInstructions,
	pages: [
		`<div class="instruction-text">
                <h1>Welcome!</h1>
                <p>Thank you for participating in this study.</p>
                <p>This experiment will take approximately 10 minutes to complete.</p>
                <p>Please make sure you:</p>
                <ul>
                    <li>Are in a quiet environment</li>
                    <li>Will not be interrupted</li>
                    <li>Can use a physical keyboard (not touchscreen)</li>
                    <li>Have your sound on</li>
                </ul>
                <p>Click 'Next' to continue.</p>
            </div>`,
	],
	show_clickable_nav: true,
};

// Instructions
const instructions = {
	type: jsPsychInstructions,
	pages: function () {
		const size = experimentState.matrixSize;
		const keyElements = KEY_MAPPINGS[size]
			.map((k) => `<span class="inline-key">${k}</span>`)
			.join(" ");

		return [
			`<div class="instruction-text">
                    <h2>Instructions</h2>
                    <p>In this task, you will see ${size} boxes on the screen.</p>
                    <p>On each trial, a mole <img src="mole.png" class="mole-image" alt="mole" style="vertical-align: middle;"> will appear in one of the boxes.</p>
                    <p>Your job is to press the corresponding key as quickly and accurately as possible.</p>
                    <p><strong>The keys you'll use are: ${keyElements}</strong></p>
                    <p>Try to respond as fast as you can while staying accurate.</p>
                </div>`,

			`<div class="instruction-text">
                    <h2>Key Mapping</h2>
                    <p>Here's which key corresponds to each position:</p>
                    <div class="key-mapping">
                        ${Array.from(
													{length: size},
													(_, i) =>
														`<div class="key-mapping-item">${KEY_MAPPINGS[size][i].toUpperCase()}</div>`,
												).join("")}
                    </div>
                    <p>Rest your fingers on these keys throughout the task.</p>
                    <p><strong>Tip:</strong> The positions correspond to a natural left-to-right hand position on the keyboard.</p>
                </div>`,

			`<div class="instruction-text">
                    <h2>Feedback</h2>
                    <p>After each response, you'll see brief feedback:</p>
                    <ul>
                        <li><span style="color: #4CAF50;">✓ Green checkmark</span> = Correct response</li>
                        <li><span style="color: #f44336;">✗ Red X + beep + "Try again!"</span> = Incorrect response</li>
                    </ul>
                    <p>If you make an error, the task will pause briefly, then automatically continue.</p>
                    <p>Just try to stay focused and respond as quickly as possible!</p>
                </div>`,

			`<div class="instruction-text">
                    <h2>Practice</h2>
                    <p>You'll start with ${CONFIG.practice_trials} practice trials to get familiar with the task.</p>
                    <p>After that, you'll complete ${CONFIG.n_blocks} blocks of trials.</p>
                    <p>Between blocks, you'll get a 15-second break to rest.</p>
                    <p>The entire task takes about 25 minutes.</p>
                    <p><strong>Ready to practice?</strong></p>
                </div>`,
		];
	},
	show_clickable_nav: true,
};

// Practice trials
function createPracticeTrial(position, trialIndex) {
	const size = experimentState.matrixSize;
	const correctKey = KEY_MAPPINGS[size][position];

	// Correction loop - repeats until correct response
	const correctionLoop = {
		timeline: [
			{
				// Practice stimulus
				type: jsPsychHtmlKeyboardResponse,
				stimulus: function () {
					return createStimulusDisplay(position, size, true);
				},
				choices: "ALL_KEYS",
				data: {
					phase: "practice",
					trial_index: trialIndex,
					position: position,
					correct_key: correctKey,
					entropy: experimentState.positionEntropies[position],
				},
				on_load: function () {
					setupKeyPressHandlers(size);
				},
				on_finish: function (data) {
					data.correct = data.response === correctKey;
				},
			},
			{
				// Feedback
				type: jsPsychHtmlKeyboardResponse,
				stimulus: function () {
					const lastTrial = jsPsych.data.get().last(1).values()[0];
					let feedbackHTML = "";
					if (lastTrial.correct) {
						feedbackHTML = '<div class="feedback-correct">✓</div>';
					} else {
						feedbackHTML = '<div class="feedback-error">✗<br>Try again!</div>';
					}
					return createStimulusDisplay(position, size, true, feedbackHTML);
				},
				choices: "NO_KEYS",
				trial_duration: function () {
					const lastTrial = jsPsych.data.get().last(1).values()[0];
					return lastTrial.correct
						? CONFIG.correct_feedback_duration
						: CONFIG.error_feedback_duration;
				},
				on_start: function () {
					const lastTrial = jsPsych.data.get().last(1).values()[0];
					if (!lastTrial.correct) {
						playErrorTone();
					}
				},
			},
		],
		loop_function: function () {
			const lastTrial = jsPsych.data.get().last(2).values()[0]; // Get the stimulus trial (2 back because feedback is last)
			return !lastTrial.correct; // Loop if incorrect
		},
	};

	// RSI after correct response
	const rsi = {
		type: jsPsychHtmlKeyboardResponse,
		stimulus: function () {
			return createStimulusDisplay(null, size, true);
		},
		choices: "NO_KEYS",
		trial_duration: CONFIG.rsi,
	};

	return {
		timeline: [correctionLoop, rsi],
	};
}

// Main task trial with feedback
function createMainTrial(position, blockNum, trialInBlock, overallTrial) {
	const size = experimentState.matrixSize;
	const correctKey = KEY_MAPPINGS[size][position];

	// Track whether an error has occurred
	let hasError = false;

	// Correction loop - repeats until correct response
	const correctionLoop = {
		timeline: [
			{
				// Main stimulus
				type: jsPsychHtmlKeyboardResponse,
				stimulus: function () {
					// Show keys if there was an error, otherwise hide them
					return createStimulusDisplay(position, size, hasError);
				},
				choices: "ALL_KEYS",
				data: {
					phase: "main",
					block: blockNum,
					trial_in_block: trialInBlock,
					overall_trial: overallTrial,
					position: position,
					correct_key: correctKey,
					entropy: experimentState.positionEntropies[position],
					matrix_size: size,
				},
				on_load: function () {
					setupKeyPressHandlers(size);
				},
				on_finish: function (data) {
					data.correct = data.response === correctKey;

					// Store in experiment state (only first response per trial)
					const trialsAtThisPosition = experimentState.trialData.filter(
						(t) => t.block === blockNum && t.trial === trialInBlock,
					);
					if (trialsAtThisPosition.length === 0) {
						experimentState.trialData.push({
							block: blockNum,
							trial: trialInBlock,
							position: position,
							response: data.response,
							rt: data.rt,
							correct: data.correct,
							entropy: data.entropy,
						});
					}
				},
			},
			{
				// Feedback
				type: jsPsychHtmlKeyboardResponse,
				stimulus: function () {
					const lastTrial = jsPsych.data.get().last(1).values()[0];
					let feedbackHTML = "";
					if (lastTrial.correct) {
						feedbackHTML = '<div class="feedback-correct">✓</div>';
						hasError = false; // Reset error flag on correct response
					} else {
						feedbackHTML = '<div class="feedback-error">✗<br>Try again!</div>';
						hasError = true; // Set error flag
					}
					// Show keys in feedback if there's an error
					return createStimulusDisplay(position, size, !lastTrial.correct, feedbackHTML);
				},
				choices: "NO_KEYS",
				trial_duration: function () {
					const lastTrial = jsPsych.data.get().last(1).values()[0];
					return lastTrial.correct
						? CONFIG.correct_feedback_duration
						: CONFIG.error_feedback_duration;
				},
				on_start: function () {
					const lastTrial = jsPsych.data.get().last(1).values()[0];
					if (!lastTrial.correct) {
						playErrorTone();
					}
				},
			},
		],
		loop_function: function () {
			const lastTrial = jsPsych.data.get().last(2).values()[0]; // Get the stimulus trial (2 back because feedback is last)
			return !lastTrial.correct; // Loop if incorrect
		},
	};

	// RSI after correct response
	const rsi = {
		type: jsPsychHtmlKeyboardResponse,
		stimulus: function () {
			return createStimulusDisplay(null, size, false);
		},
		choices: "NO_KEYS",
		trial_duration: CONFIG.rsi,
	};

	return {
		timeline: [correctionLoop, rsi],
	};
}

// Block break with feedback
function createBlockBreak(blockNum) {
	return {
		type: jsPsychHtmlKeyboardResponse,
		stimulus: function () {
			// Calculate block statistics
			const blockTrials = experimentState.trialData.filter((t) => t.block === blockNum - 1);
			const accuracy = (
				(blockTrials.filter((t) => t.correct).length / blockTrials.length) *
				100
			).toFixed(1);
			const meanRT = (
				blockTrials.filter((t) => t.correct).reduce((sum, t) => sum + t.rt, 0) /
				blockTrials.filter((t) => t.correct).length
			).toFixed(0);

			// Get previous block stats if available
			let previousStats = "";
			if (blockNum > 1) {
				const prevBlockTrials = experimentState.trialData.filter((t) => t.block === blockNum - 2);
				const prevAccuracy = (
					(prevBlockTrials.filter((t) => t.correct).length / prevBlockTrials.length) *
					100
				).toFixed(1);
				const prevMeanRT = (
					prevBlockTrials.filter((t) => t.correct).reduce((sum, t) => sum + t.rt, 0) /
					prevBlockTrials.filter((t) => t.correct).length
				).toFixed(0);

				previousStats = `
                        <p><strong>Previous Block:</strong></p>
                        <p>Accuracy: ${prevAccuracy}% | Mean RT: ${prevMeanRT}ms</p>
                    `;
			}

			// Adaptive feedback
			let feedback = "";
			if (accuracy < 85) {
				feedback = '<p style="color: #f44336;"><strong>Try to be more accurate.</strong></p>';
			} else if (meanRT > 500) {
				feedback = '<p style="color: #2196F3;"><strong>Try to respond faster!</strong></p>';
			} else {
				feedback = '<p style="color: #4CAF50;"><strong>Great job! Keep it up!</strong></p>';
			}

			const progress = ((blockNum / CONFIG.n_blocks) * 100).toFixed(0);

			return `
                    <div class="block-feedback">
                        <h2>Block ${blockNum} of ${CONFIG.n_blocks} Complete!</h2>
                        
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: ${progress}%"></div>
                        </div>
                        
                        <p><strong>Current Block:</strong></p>
                        <p>Accuracy: ${accuracy}% | Mean RT: ${meanRT}ms</p>
                        
                        ${previousStats}
                        
                        ${feedback}
                        
                        <p style="margin-top: 30px;">Take a 15-second break.</p>
                        <p style="font-size: 14px; color: #666;">The next block will start automatically.</p>
                    </div>
                `;
		},
		choices: "NO_KEYS",
		trial_duration: CONFIG.block_break_duration,
	};
}

// Post-task questionnaire
const questionnaire = {
	type: jsPsychSurveyMultiChoice,
	questions: [
		{
			prompt: "Did you notice anything special about the task?",
			name: "noticed_special",
			options: ["No", "Yes"],
			required: true,
		},
	],
};

const awareness_questions = {
	timeline: [
		{
			type: jsPsychSurveyText,
			questions: [
				{
					prompt: "What did you notice? Please describe in detail.",
					name: "what_noticed",
					rows: 4,
					required: true,
				},
			],
		},
	],
	conditional_function: function () {
		const lastResponse = jsPsych.data.get().last(1).values()[0];
		return lastResponse.response.noticed_special === "Yes";
	},
};

const pattern_question = {
	type: jsPsychSurveyMultiChoice,
	questions: [
		{
			prompt: "Did you notice any regularity or pattern in which positions the mole appeared?",
			name: "noticed_pattern",
			options: ["No", "Yes"],
			required: true,
		},
	],
};

const pattern_description = {
	timeline: [
		{
			type: jsPsychSurveyText,
			questions: [
				{
					prompt: "Can you describe the pattern?",
					name: "pattern_description",
					rows: 4,
					required: true,
				},
			],
		},
	],
	conditional_function: function () {
		const lastResponse = jsPsych.data.get().last(1).values()[0];
		return lastResponse.response.noticed_pattern === "Yes";
	},
};

const strategy_question = {
	type: jsPsychSurveyText,
	questions: [
		{
			prompt: "Did you use any strategy to help you respond faster? If so, please describe.",
			name: "strategy",
			rows: 4,
			required: false,
		},
	],
};

const forced_pattern_question = {
	type: jsPsychSurveyText,
	questions: [
		{
			prompt: "There WAS a regularity in the sequence. What do you think it was?",
			name: "forced_pattern",
			rows: 4,
			required: true,
		},
	],
};

// Debrief
const debrief = {
	type: jsPsychInstructions,
	pages: [
		`<div class="instruction-text">
                <h2>Thank You!</h2>
                <p>You have completed the experiment.</p>
                <p>This study investigates how people learn predictable patterns in sequences.</p>
                <p>Some positions had the mole appearing in more predictable locations,
                while others were more variable.</p>
                <p>Your data will help us understand how people learn these kinds of patterns.</p>
                <p>Thank you for your participation!</p>
                <p>You may close this window now.</p>
            </div>`,
	],
	show_clickable_nav: true,
};

// Initialize experiment
initializeExperiment();

// Add components to timeline
timeline.push(welcome);
timeline.push(instructions);

// Practice block
const practiceSequence = generateSequence(experimentState.transitionMatrix, CONFIG.practice_trials);

for (let i = 0; i < CONFIG.practice_trials; i++) {
	timeline.push(createPracticeTrial(practiceSequence[i], i));
}

// Practice feedback
timeline.push({
	type: jsPsychHtmlButtonResponse,
	stimulus: function () {
		const practiceData = jsPsych.data.get().filter({phase: "practice"});
		const accuracy = (
			(practiceData.filter({correct: true}).count() / practiceData.count()) *
			100
		).toFixed(1);

		return `
                <div class="instruction-text">
                    <h2>Practice Complete!</h2>
                    <p>Your accuracy: ${accuracy}%</p>
                    <p>Remember: Respond as quickly and accurately as possible.</p>
                    <p>The main task will now begin.</p>
                </div>
            `;
	},
	choices: ["Start Main Task"],
});

// Main task blocks
for (let block = 0; block < CONFIG.n_blocks; block++) {
	for (let trial = 0; trial < CONFIG.trials_per_block; trial++) {
		const overallTrial = block * CONFIG.trials_per_block + trial;
		const position = experimentState.sequence[overallTrial];

		timeline.push(createMainTrial(position, block, trial, overallTrial));
	}

	// Block break (except after last block)
	if (block < CONFIG.n_blocks - 1) {
		timeline.push(createBlockBreak(block + 1));
	}
}

// Post-task questionnaire
timeline.push(questionnaire);
timeline.push(awareness_questions);
timeline.push(pattern_question);
timeline.push(pattern_description);
timeline.push(strategy_question);
timeline.push(forced_pattern_question);

// Debrief
timeline.push(debrief);

jsPsych.run(timeline);
