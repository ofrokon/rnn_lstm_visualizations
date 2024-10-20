# Recurrent Neural Networks and LSTMs: Understanding Sequential Data with PyTorch

This repository contains Python scripts demonstrating the implementation and visualization of Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks using PyTorch. It accompanies the Medium post "Recurrent Neural Networks and LSTMs: Understanding Sequential Data with PyTorch".

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Visualizations](#visualizations)
4. [Topics Covered](#topics-covered)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run these scripts, you need Python 3.6 or later. Follow these steps to set up your environment:

1. Clone this repository:
   ```
   git clone https://github.com/ofrokon/rnn-lstm-sequential-data.git
   cd rnn-lstm-sequential-data
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To generate all visualizations and train the RNN and LSTM models, run:

```
python rnn_lstm_visualizations.py
```

This will create PNG files for visualizations and print training progress in the console.

## Visualizations

This script generates the following visualizations:

1. `rnn_architecture.png`: Diagram of RNN architecture
2. `lstm_architecture.png`: Diagram of LSTM architecture
3. `training_loss.png`: Training loss comparison between RNN and LSTM
4. `prediction_comparison.png`: Sequence prediction comparison between RNN and LSTM

## Topics Covered

1. RNN Architecture
2. LSTM Architecture
3. Implementing RNN and LSTM with PyTorch
4. Training RNN and LSTM models
5. Comparing RNN and LSTM performance on sequence prediction

Each topic is explained in detail in the accompanying Medium post, including Python implementation and visualizations.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your suggested changes. If you're planning to make significant changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).

---

For a detailed explanation of RNN and LSTM concepts and their implementation using PyTorch, check out the accompanying Medium post: [Recurrent Neural Networks and LSTMs: Understanding Sequential Data with PyTorch](https://medium.com/@mroko001/rnns-and-lstms-understanding-sequential-data-with-pytorch-b8409e706f0c)

For questions or feedback, please open an issue in this repository.
