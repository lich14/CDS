REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_sd_agent import RNN_SD_Agent

REGISTRY["rnn"] = RNN_SD_Agent
