"""
Benchmark script for VAE forward pass with different configurations.

Compares the two combinations of:
- use_torch_distributions: True/False (affects both sampling and KL calculation)

Across varying:
- hidden_dim
- latent_dim
- batch_size
- n_samples
"""

import time
from itertools import product

import pandas as pd
import torch
from tqdm import tqdm

from model import VAE, VAEConfig


def benchmark_forward_pass(
    model: VAE,
    batch_size: int,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Benchmark a single configuration of the VAE forward pass.

    Args:
        model: VAE model instance
        batch_size: Batch size for input
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        device: Device to run on

    Returns:
        Dictionary with timing statistics
    """
    model.eval()

    # Create dummy input
    x = torch.randn(batch_size, 1, 28, 28, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x, compute_loss=True)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x, compute_loss=True)
            end = time.perf_counter()
            times.append(end - start)

    times = torch.tensor(times)

    return {
        "mean_ms": times.mean().item() * 1000,
        "std_ms": times.std().item() * 1000,
        "min_ms": times.min().item() * 1000,
        "max_ms": times.max().item() * 1000,
        "median_ms": times.median().item() * 1000,
    }


def run_benchmark_suite(
    hidden_dims: list[int],
    latent_dims: list[int],
    batch_sizes: list[int],
    n_samples_list: list[int],
    input_dim: int = 784,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run comprehensive benchmark suite across all configurations.

    Args:
        hidden_dims: List of hidden dimensions to test
        latent_dims: List of latent dimensions to test
        batch_sizes: List of batch sizes to test
        n_samples_list: List of sample counts to test
        input_dim: Input dimension (default 784 for MNIST)
        num_warmup: Number of warmup iterations per config
        num_runs: Number of benchmark iterations per config
        device: Device to run on

    Returns:
        DataFrame with all benchmark results
    """
    results = []

    # All combinations of settings
    configs = list(
        product(
            hidden_dims,
            latent_dims,
            batch_sizes,
            n_samples_list,
            [True, False],  # use_torch_distributions
        )
    )

    total = len(configs)

    with tqdm(total=total, desc="Benchmarking") as pbar:
        for (
            hidden_dim,
            latent_dim,
            batch_size,
            n_samples,
            use_torch_distributions,
        ) in configs:
            # Create model for this configuration
            config = VAEConfig(
                input_dim,
                hidden_dim,
                latent_dim,
                use_torch_distributions=use_torch_distributions,
                n_samples=n_samples,
            )
            model = VAE(config).to(device)

            # Run benchmark
            timing_stats = benchmark_forward_pass(
                model=model,
                batch_size=batch_size,
                num_warmup=num_warmup,
                num_runs=num_runs,
                device=device,
            )

            # Store results
            result = {
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "batch_size": batch_size,
                "n_samples": n_samples,
                "use_torch_distributions": use_torch_distributions,
                **timing_stats,
            }
            results.append(result)

            pbar.update(1)
            pbar.set_postfix(
                {
                    "h": hidden_dim,
                    "l": latent_dim,
                    "b": batch_size,
                    "n": n_samples,
                    "dist": use_torch_distributions,
                    "t": f"{timing_stats['mean_ms']:.2f}ms",
                }
            )

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print summary statistics grouped by configuration."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group by the boolean flags and show average across all other parameters
    print("\n--- Average performance by configuration (across all sizes) ---")
    summary = (
        df.groupby(["use_torch_distributions"])
        .agg(
            {
                "mean_ms": ["mean", "std"],
                "median_ms": "mean",
            }
        )
        .round(3)
    )
    print(summary)

    # Find best configuration overall
    print("\n--- Best configuration (lowest average time) ---")
    best_config = df.groupby(["use_torch_distributions"])["mean_ms"].mean()
    best = best_config.idxmin()
    print(f"use_torch_distributions={best}")
    print(f"Average time: {best_config[best]:.3f} ms")

    # Breakdown by dimension type
    print("\n--- Performance vs Hidden Dimension ---")
    hidden_analysis = (
        df.groupby(["hidden_dim", "use_torch_distributions"])["mean_ms"]
        .mean()
        .unstack([1])
    )
    print(hidden_analysis.round(3))

    print("\n--- Performance vs Latent Dimension ---")
    latent_analysis = (
        df.groupby(["latent_dim", "use_torch_distributions"])["mean_ms"]
        .mean()
        .unstack([1])
    )
    print(latent_analysis.round(3))

    print("\n--- Performance vs Batch Size ---")
    batch_analysis = (
        df.groupby(["batch_size", "use_torch_distributions"])["mean_ms"]
        .mean()
        .unstack([1])
    )
    print(batch_analysis.round(3))

    print("\n--- Performance vs Number of Samples ---")
    samples_analysis = (
        df.groupby(["n_samples", "use_torch_distributions"])["mean_ms"]
        .mean()
        .unstack([1])
    )
    print(samples_analysis.round(3))


def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark VAE configurations")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Hidden dimensions to test",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[2, 8, 16, 32, 64],
        help="Latent dimensions to test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        nargs="+",
        default=[1],
        help="Number of samples to test",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-runs", type=int, default=1000, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    print("Starting VAE Benchmark Suite")
    print(f"Hidden dimensions: {args.hidden_dims}")
    print(f"Latent dimensions: {args.latent_dims}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark runs: {args.num_runs}")
    print(f"Device: {args.device}")
    print()

    # Run benchmark suite
    df = run_benchmark_suite(
        hidden_dims=args.hidden_dims,
        latent_dims=args.latent_dims,
        batch_sizes=args.batch_sizes,
        n_samples_list=args.n_samples,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device=args.device,
    )

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
