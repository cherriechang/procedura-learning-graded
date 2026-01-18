# Procedural Learning Experiment

A web-based procedural learning experiment using jsPsych's Serial Reaction Time (SRT) task to measure participants' ability to learn sequential patterns.

## Overview

This experiment implements a Serial Reaction Time task, which is a classic paradigm for studying procedural learning. Participants respond to visual targets appearing in one of four locations by pressing corresponding keys. The task includes:

- **Practice trials**: 8 random trials to familiarize participants with the task
- **Main task**: 5 blocks of 32 trials each (160 trials total)
- **Implicit learning**: A repeating 8-item sequence pattern to test procedural learning
- **Scoring system**: Combines accuracy and response speed to produce a procedural learning score

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/cherriechang/procedural-learning.git
   cd procedural-learning
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Experiment

Start a local web server:
```bash
npm start
```

Then open your browser and navigate to:
```
http://localhost:8080
```

## How the Task Works

### Instructions for Participants

- Participants see a grid with 4 boxes arranged horizontally
- A target (filled box) appears in one location at a time
- Participants must press the corresponding key as quickly and accurately as possible:
  - **Z** = leftmost position
  - **X** = second position
  - **C** = third position
  - **V** = rightmost position

### Experimental Design

The experiment uses a repeating sequence pattern: `[0, 2, 1, 3, 2, 0, 3, 1]`

This pattern repeats 4 times per block across 5 blocks, allowing measurement of:
- **Explicit learning**: Conscious awareness of the pattern
- **Implicit learning**: Unconscious improvement in reaction time
- **Procedural memory**: Ability to learn motor sequences

### Scoring

At the end of the experiment, participants receive:

1. **Procedural Learning Score**: A composite score calculated from accuracy and reaction time
   - Formula: `(Accuracy% × 100) / (Average RT / 10)`
   - Higher scores indicate better procedural learning ability

2. **Accuracy percentage**: Percentage of correct responses
3. **Average reaction time**: Mean response time for correct trials (in milliseconds)

## File Structure

```
procedural-learning/
├── index.html          # Main HTML file with experiment interface
├── experiment.js       # JavaScript code implementing the SRT task
├── package.json        # Node.js dependencies and scripts
└── README.md          # This file
```

## Dependencies

- **jsPsych 7.3.4**: Core experiment framework
- **@jspsych/plugin-html-keyboard-response**: For displaying instructions and feedback
- **@jspsych/plugin-serial-reaction-time**: For the SRT task itself
- **@jspsych/plugin-instructions**: For multi-page instructions
- **http-server**: Development server for local testing

## Data Collection

The experiment tracks the following data for each trial:
- Target location
- Participant response
- Reaction time
- Accuracy (correct/incorrect)
- Block number
- Trial phase (practice/main)

All data is available through jsPsych's data collection methods and can be extended to save to a server or download as CSV/JSON.

## Customization

You can modify the following parameters in `experiment.js`:

- `sequence_pattern`: The repeating sequence (line 80)
- `num_blocks`: Number of experimental blocks (line 154)
- `num_repetitions`: How many times the sequence repeats per block (line 155)
- Scoring formula in the `on_finish` callback (line 16)

## Scientific Background

The Serial Reaction Time task was developed by Nissen & Bullemer (1987) and is widely used in cognitive neuroscience to study:
- Procedural learning and memory
- Implicit vs. explicit learning
- Motor sequence learning
- Skill acquisition
- Comparisons with declarative memory tasks

## License

MIT

## Citation

If you use this experiment in your research, please cite:

Nissen, M. J., & Bullemer, P. (1987). Attentional requirements of learning: Evidence from performance measures. Cognitive Psychology, 19(1), 1-32.
