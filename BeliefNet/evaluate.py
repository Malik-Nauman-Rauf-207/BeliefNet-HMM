from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py

if os.name == "nt":
    # Prevent @torch.compile decorators in ss modules from invoking Inductor on Windows.
    torch.compile = lambda fn=None, *args, **kwargs: (
        (lambda f: f) if fn is None else fn
    )

from ss.utility.learning.process.config import TestingConfig
from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.process import LearningHmmFilterProcess
from ss.utility.learning.compile import CompileContext

def reconstruct_loaders_from_hdf5(data_file,
                                  data_split):
    """
    Reconstruct eval_loader from saved HDF5 file.
    
    Args:
        data_file: Path to the HDF5 file containing the datasets
        data_split: Which dataset to reconstruct ("z_train", "z_validation", or "z_test")
    
    Returns:
        eval_loader: Reconstructed PyTorch DataLoader
    """
    
    with h5py.File(data_file, 'r') as f:
        eval_inputs_tensor = torch.tensor(f[data_split][:,:-1]).unsqueeze(1)
        eval_targets_tensor = torch.tensor(f[data_split][:,1:]).unsqueeze(1)

    # Create TensorDatasets
    eval_dataset = TensorDataset(eval_inputs_tensor, eval_targets_tensor)
    
    # Create DataLoaders
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    print(f"Reconstructed train_loader: {len(eval_dataset)} samples, {len(eval_loader)} batches")
    
    return eval_loader

def evaluate_loss(
    model_folder_path,
    model_file_name,
    evaluation_dataset,
    max_batch=None,
):
    from ss.utility.learning.parameter.transformer.softmax import SoftmaxTransformer
    from ss.utility.learning.parameter.transformer.softmax.config import SoftmaxTransformerConfig

    # Load the model
    model_filepath = Path(model_folder_path) / model_file_name
    learning_filter, _ = LearningHmmFilter[
        SoftmaxTransformer, SoftmaxTransformerConfig
    ].load(
        model_filepath,
        safe_callables={
            torch.nn.functional.cross_entropy,
            torch.optim.AdamW,
            # types of extra arguments
        },
        strict=False,
    )
    learning_filter.eval()

    evaluation_process = LearningHmmFilterProcess(
        module=learning_filter,
        loss_function=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.AdamW(learning_filter.parameters(), lr=1e-3),
    )
    testing_config = TestingConfig(max_batch=max_batch)

    with torch.inference_mode():
        with CompileContext():
            losses = evaluation_process.evaluate_model(
                evaluation_dataset,
                testing_config,
            )
    return losses

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "../prepare_data/data_d64_m128_lamda0.9_example.h5"
    model_folder_path = base_dir / "Model_m128/full_matrix/cpu_char_i2000_b10_dr0.1_d64"
    model_file_name = "learning_filter.pt"
    data_split = "z_validation"
    # data_file = base_dir / "../prepare_data/data_d64_m32_lamda0.9_example.h5"
    # model_folder_path = base_dir / "Model_m32_example/full_matrix/cpu_char_i2000_b10_dr0.1_d64"# edit this
    # model_file_name = "learning_filter.pt"#
    # data_split = "z_test"  # "z_train", "z_validation", or "z_test" 

    # Reconstruct evaluation dataset
    evaluation_dataset = reconstruct_loaders_from_hdf5(data_file, data_split)
    
    # Evaluate the model and print losses
    losses = evaluate_loss(
        model_folder_path,
        model_file_name,
        evaluation_dataset,
        max_batch=10000000,
    )
    avg_loss = losses.sum() / len(losses)
    print("Evaluation loss:", avg_loss)