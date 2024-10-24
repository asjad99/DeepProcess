# DeepProcess Experiment Codes

This repository contains experiment codes for **DeepProcess**, an implementation aimed at investigating sequence modeling and deep neural network-based process prediction. The files included here support experiments on sequence-to-sequence models, Dynamic Neural Computer (DNC), and auxiliary modules for experimentation.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Main Components](#main-components)
  - [Baseline Sequence-to-Sequence Model](#baseline-sequence-to-sequence-model)
  - [Prefix-Suffix Training and Execution](#prefix-suffix-training-and-execution)
  - [Dynamic Neural Computer Modules](#dynamic-neural-computer-modules)
- [License](#license)

## Overview

The **DeepProcess** project is designed to explore how deep learning models can be applied to process prediction problems. The implementation includes experiments with:

- Sequence-to-Sequence (Seq2Seq) models for baseline analysis.
- Prefix-Suffix models to predict sequences based on partial data.
- The Dynamic Neural Computer (DNC) for advanced memory and processing tasks.

## Project Structure

The main experiment codes are organized as follows:

```
- baseline_seq2seq.py       # Baseline Seq2Seq model implementation
- presuf_train.py          # Training script for prefix-suffix models
- presuf_run.py            # Execution script for evaluating prefix-suffix models
- controller.py            # Controller class used in DNC
- dnc.py                   # Dynamic Neural Computer implementation
- feedforward_controller.py # Feedforward Controller for DNC
- recurrent_controller.py  # Recurrent Controller for DNC
- memory.py                # Memory module for DNC
- seq_helper.py            # Helper functions for sequence operations
- utility.py               # Utility functions used throughout the project
```

## Installation

To set up the environment for running these experiments, please follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/asjad99/DeepProcess
   cd DeepProcess/deep_process_experiment_codes
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv deepprocess-env
   source deepprocess-env/bin/activate   # On Windows: deepprocess-env\Scripts\activate
   ```

3. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

### How to Run Experiments
1. Data for each experiment can be found in the `./data/BusinessProcess` folder.
2. The file `presuf_run.py` contains code for 3 experiments.
3. In `presuf_run.py`, there are train and test functions for each task. Just call the appropriate one based on your requirement.

### Baseline Seq2Seq Model

To train the baseline sequence-to-sequence model, use the following command:

```sh
python baseline_seq2seq.py
```

This script provides a foundational sequence-to-sequence model for comparison with other, more advanced models in the repository.

## How to Tune Hyper Parameters

1. In each function, hyperparameters are hard-coded.
2. You can directly edit the hyperparameters in the function definitions to change their values.

### Type of Hyper Parameters

1. **Method Type** (edit in constructor's arguments):
   - **LSTM Seq2Seq**: `use_mem=False`
   - **DNC**: `use_mem=True`, `decoder_mode=True/False`, `dual_controller=False`, `write_protect=False`
   - **DC-MANN**: `use_mem=True`, `decoder_mode=True`, `dual_controller=True`, `write_protect=False`
   - **DCw_MANN**: `use_mem=True`, `decoder_mode=True/False`, `dual_controller=True`, `write_protect=True`

2. **Model Parameters** (edit in constructor's arguments):
   - `use_emb=True/False`: Use embedding layer or not.
   - `dual_emb=True/False`: If using embedding layer, use one shared or two embeddings for encoder and decoder.
   - `hidden_controller_dim`: Dimension of controller hidden state.

3. **Memory Parameters** (if using memory):
   - `words_count`: Number of memory slots.
   - `word_size`: Size of each memory slot.
   - `read_heads`: Number of reading heads.

4. **Training Parameters**:
   - `batch_size`: Number of sequences sampled per batch.
   - `iterations`: Maximum number of training steps.
   - `lm_train=True/False`: Training by the language model's way (edit in `prepare_sample_batch` function).
   - **Optimizer**: In file `dnc.py`, function `build_loss_function_mask` (default is Adam).

#### Notes

1. The current hyperparameters are picked based on experience from other projects.
2. Except for different method types, other hyperparameter combinations have not been extensively tested.

## Main Components

### Baseline Sequence-to-Sequence Model

- **File**: `baseline_seq2seq.py`
- **Description**: Implements a standard Seq2Seq model to serve as a baseline for comparing more advanced sequence models. This script includes basic encoder-decoder architecture and attention mechanisms.

### Prefix-Suffix Training and Execution

- **Files**: `presuf_train.py`, `presuf_run.py`
- **Description**: The `presuf_train.py` script is used for training models that predict suffixes from given prefixes. The `presuf_run.py` script is then used to evaluate these models on different datasets. These scripts help in capturing meaningful sequence predictions from partial sequences.

### Dynamic Neural Computer Modules

The implementation of a **Dynamic Neural Computer (DNC)** is broken down into several components:

- **Controller Modules**:

  - `controller.py`: Base controller class for managing the learning process.
  - `feedforward_controller.py`: Feedforward controller used to manipulate the memory.
  - `recurrent_controller.py`: Recurrent controller that adds sequence-dependent memory updates.

- **Memory Management**:

  - `memory.py`: Implements the memory module responsible for reading and writing data for the DNC.

- **DNC Core**:

  - `dnc.py`: The main implementation of the Dynamic Neural Computer, including operations for managing external memory and interacting with controllers.

- **Helper Functions**:

  - `seq_helper.py`: Utility functions that assist with sequence preprocessing and other related tasks.
  - `utility.py`: General-purpose utility functions used throughout the project.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---
Feel free to explore the individual scripts to understand their contributions to the overall sequence modeling experiments. If you have any questions or suggestions, please open an issue or submit a pull request!

