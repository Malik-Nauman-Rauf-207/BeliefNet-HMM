import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import h5py as h5

# Prevent BLAS/OpenMP thread deadlocks across parallel processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from hmmlearn import hmm

def evaluate_single_fit(idx, seed_offset, n_iter, d, m, X, Y, x_lengths, y_lengths):
    start = time.time()
    print(f"[*] Starting Fit #{idx + 1} (Seed: {idx + seed_offset}, n_iter: {n_iter})...", flush=True)

    model = hmm.CategoricalHMM(
        n_components=d,
        random_state=idx + seed_offset,
        init_params='ste',
        verbose=False,
        n_iter=n_iter
    )

    # Train via Baum-Welch EM
    model.fit(X.reshape(-1, 1), x_lengths)
    elapsed = time.time() - start

    # Per-token Cross-Entropy / Negative Log-Likelihood
    train_loss = -model.score(X.reshape(-1, 1), x_lengths) / X.size
    eval_loss = -model.score(Y.reshape(-1, 1), y_lengths) / Y.size

    print(f"[+] Finished Fit #{idx + 1} in {elapsed:.1f}s | Val Loss: {eval_loss:.4f} | Train Loss: {train_loss:.4f}", flush=True)
    return {
        "idx": idx,
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "time": elapsed,
        "A": model.transmat_,
        "C": model.emissionprob_,
        "pi": model.startprob_
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="../prepare_data/data_d64_m128_lamda0.9_example.h5")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_fits", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--seed_offset", type=int, default=123)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        data_path = Path("..") / args.data_file

    with h5.File(data_path, "r") as f:
        training_set = f["z_train"][:]
        validation_set = f["z_validation"][:]

    m = int(training_set.max() + 1)
    X = training_set.flatten()
    Y = validation_set.flatten()
    x_lengths = [training_set.shape[1]] * training_set.shape[0]
    y_lengths = [validation_set.shape[1]] * validation_set.shape[0]

    print("\n" + "=" * 60)
    print(f"Running Baum-Welch (EM) Benchmark Replication")
    print(f"Data: d={args.d}, m={m} | {args.n_fits} fits, {args.n_iter} iters | Workers: {args.workers}")
    print("=" * 60 + "\n", flush=True)

    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_fit,
                idx, args.seed_offset, args.n_iter, args.d, m,
                X, Y, x_lengths, y_lengths
            )
            for idx in range(args.n_fits)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    total_time = time.time() - t0
    best_result = min(results, key=lambda r: r["eval_loss"])

    print("\n" + "=" * 60)
    print("              BAUM-WELCH REPLICATION RESULTS")
    print("=" * 60)
    print(f"Best Validation Loss:        {best_result['eval_loss']:.4f}")
    print(f"Paper Target (Table 1):      1.951")
    print(f"Optimal Filter Bound:        1.368")
    print(f"Total Wall-Clock Time:       {total_time:.1f}s ({total_time / 60:.2f} min)")
    print("=" * 60 + "\n")

    # Save the best model
    out_file = f"bw_best_d{args.d}_m{m}.h5"
    with h5.File(out_file, "w") as f_out:
        f_out.create_dataset("A", data=best_result["A"])
        f_out.create_dataset("C", data=best_result["C"])
        f_out.create_dataset("pi", data=best_result["pi"])
        f_out.create_dataset("eval_loss", data=best_result["eval_loss"])
    print(f"[+] Saved best model matrices to {out_file}")