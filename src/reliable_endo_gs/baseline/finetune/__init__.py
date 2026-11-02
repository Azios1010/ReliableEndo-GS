"""Full-grid fine-tuning of posed feed-forward Gaussian baselines on the A00 train split.

Integration layer only: upstream imports stay lazy and nothing here initializes CUDA or reads a dataset at import time.
"""
