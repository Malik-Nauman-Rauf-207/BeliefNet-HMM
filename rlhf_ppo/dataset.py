# rlhf_ppo/dataset.py
import torch
import h5py
import os

class HDF5HMMDataset:
    def __init__(self, h5_path="../prepare_data/", seq_len=64, key='z_train'):
        """
        Loads HMM sequence data directly from .h5 files in prepare_data.
        `self.key` records the dataset key that was ACTUALLY used. If the requested key is
        missing the loader falls back to another key, so check `dataset.key` when you ask for
        a validation split (a silent fallback to the training data would hide overfitting).
        """
        if os.path.isdir(h5_path):
            h5_files = [f for f in os.listdir(h5_path) if f.endswith('.h5')]
            if not h5_files:
                raise FileNotFoundError(f"No .h5 files found in directory: {h5_path}")
            h5_path = os.path.join(h5_path, h5_files[0])

        print(f"Loading dataset from HDF5 file: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            if key not in f:
                available = [k for k in f.keys() if 'z_' in k]
                key = available[0] if available else list(f.keys())[0]
            print(f"Using dataset key: '{key}'")
            self.key = key

            # Read token sequence array
            raw_data = f[key][:]

        data_tensor = torch.tensor(raw_data, dtype=torch.long)

        # Reshape if data is a continuous 1D token stream
        if data_tensor.dim() == 1:
            num_seqs = len(data_tensor) // seq_len
            data_tensor = data_tensor[:num_seqs * seq_len].view(num_seqs, seq_len)

        # The model needs inputs of length <= block_size; the PPO script feeds seq[:, :-1].
        assert data_tensor.shape[1] <= seq_len + 1, (
            f"Sequences have length {data_tensor.shape[1]} but block_size is {seq_len}; "
            f"positional embeddings would overflow.")

        self.data = data_tensor
        print(f"Dataset successfully loaded. Total sequences: {self.data.shape[0]}, Sequence length: {self.data.shape[1]}")

    def get_batch(self, batch_size=32):
        ix = torch.randint(len(self.data), (batch_size,))
        return self.data[ix]