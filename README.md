# Procedural Learning Experiment

> **Status:** Currently in pilot testing phase

A web-based procedural learning experiment implementing a Serial Reaction Time (SRT) task with transition probability matrices to measure participants' ability to learn sequential patterns.

## Overview

This experiment implements a Serial Reaction Time task with probabilistic sequences. Participants respond to a mole appearing in one of multiple locations by pressing corresponding keys. The task includes:

- **Variable matrix size**: Randomly assigned 4, 5, 6, 7, or 8 positions per participant
- **Practice trials**: 2x the matrix size (familiarization with random sampling)
- **Main task**: 7 blocks of 10x matrix size trials each
- **Probabilistic learning**: Uses transition probability matrices with varying entropy levels
- **Adaptive feedback**: Performance-based feedback during block breaks

## Running the Experiment

### Live Demo

Try the experiment online: [https://cherriechang.github.io/procedural-learning-graded/](https://cherriechang.github.io/procedural-learning-graded/)

### Local Development

Simply open `index.html` in a web browser. No installation or build process required.

For local development with a server:
```bash
python -m http.server 8000
```

Then navigate to `http://localhost:8000`

## How the Task Works

### Instructions for Participants

- Participants see boxes arranged horizontally (4-8 boxes depending on condition)
- A mole appears in one box at a time
- Participants must press the corresponding key as quickly and accurately as possible
- Key mappings vary by condition:
  - **4 positions**: D, F, J, K
  - **5 positions**: S, D, F, J, K
  - **6 positions**: S, D, F, J, K, L
  - **7 positions**: A, S, D, F, J, K, L
  - **8 positions**: A, S, D, F, J, K, L, ;

### Experimental Design

The experiment uses transition probability matrices where each position has probabilistic transitions to the next position. Key features:

- **Random condition assignment**: Each participant is randomly assigned a matrix size (4-8 positions)
- **Shuffled matrices**: Transition matrices are shuffled to vary which positions are high vs. low entropy
- **Probabilistic sequences**: Generated on-the-fly based on transition probabilities
- **Error correction**: Participants must correct errors before continuing
- **Adaptive feedback**: Performance-based messages during self-paced breaks

### Feedback System

During block breaks, participants see:
- Current block accuracy
- Previous block accuracy (for comparison)
- Adaptive feedback based on performance:
  - If accuracy < 85%: "Try to be more accurate"
  - If mean RT > 1000ms: "Try to respond faster!"
  - Otherwise: "Great job! Keep it up!"
- Progress bar showing completion
- Self-paced continuation (no forced break duration)

## File Structure

```
procedural-learning-graded/
├── index.html          # Main HTML file
├── experiment.js       # Main experiment logic
├── styles.css          # Custom styling
├── modules/
│   └── matrices.js     # Transition probability matrices
├── assets/             # Images (mole, keys, GIFs)
└── README.md          # This file
```

## Dependencies

All dependencies are loaded via CDN (no installation required):

- **jsPsych 7.3.4**: Core experiment framework
- **@jspsych/plugin-html-keyboard-response**: For stimulus presentation and response collection
- **@jspsych/plugin-instructions**: For multi-page instructions
- **@jspsych/plugin-html-button-response**: For feedback screens and questionnaire
- **@jspsych/plugin-survey-text**: For open-ended questionnaire responses
- **@jspsych/plugin-survey-multi-choice**: For multiple choice questions
- **@jspsych/plugin-survey-likert**: For confidence ratings
- **@jspsych/plugin-preload**: For loading media files before experiment
- **@jspsych-contrib/plugin-pipe**: For data saving to DataPipe

## Data Collection

The experiment tracks the following data for each trial:
- Current position (where mole appeared)
- Participant response key
- Reaction time (ms)
- Correctness (boolean)
- Block number
- Trial number (within block and overall)
- Trial phase (practice/main)
- Error count (for trials with errors)
- Correction sequence data

Additionally, experiment-level data is recorded:
- Matrix size (4-8 positions)
- Complete transition matrix
- Full sequence generated
- Key mapping for the participant
- All experiment configuration parameters

Post-task questionnaire responses include:
- Open-ended awareness questions
- Pattern regularity detection
- Confidence ratings
- Strategy descriptions
- Forced-choice pattern descriptions

All data is automatically saved to DataPipe after the questionnaire is completed.

## Customization

Key parameters in `EXPERIMENT_CONFIG`:
- `matrix_size`: Randomly assigned (4, 5, 6, 7, or 8)
- `n_blocks`: Number of main blocks (default: 7)
- `trials_per_block`: 10x matrix size
- `practice_trials`: 2x matrix size
- `rsi`: Response-stimulus interval in ms (default: 120)
- `error_feedback_duration`: Duration of error feedback (default: 200ms)
- `block_break_duration`: Now self-paced (user clicks to continue)

## Scientific Background

The Serial Reaction Time task was developed by Nissen & Bullemer (1987) and is widely used in cognitive neuroscience to study procedural learning and memory. This implementation extends the classic paradigm by:

- Using probabilistic transition matrices instead of deterministic sequences
- Varying the number of positions (4-8) across participants
- Incorporating entropy-graded positions within each matrix
- Including comprehensive awareness measures
- Implementing error correction to ensure sequence integrity

This design allows for investigation of:
- Implicit statistical learning
- Effects of matrix complexity on learning
- Individual differences in procedural learning
- Explicit vs. implicit knowledge acquisition
- Position-specific learning based on entropy levels

## License

MIT

## Citation