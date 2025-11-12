import argparse
import json
import os
import sys
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T

from model import VAE, VAEConfig
from trainer import VAETrainer
from trainer_config import TrainerConfig


def get_dataloaders(batch_size: int):
    """Create MNIST data loaders."""
    transform = T.ToTensor()

    train_data = datasets.MNIST(
        os.path.expanduser("~/.pytorch/MNIST_data/"),
        download=True,
        train=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        os.path.expanduser("~/.pytorch/MNIST_data/"),
        download=True,
        train=False,
        transform=transform,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def parse_args():
    """Parse command-line arguments and create TrainerConfig."""
    parser = argparse.ArgumentParser(
        description="Train a Variational Autoencoder on MNIST"
    )

    # Model architecture
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        help="Dimensionality of the latent space (default: 2)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Dimensionality of the hidden layer (default: 512)",
    )
    parser.add_argument(
        "--input-shape",
        type=tuple,
        default=(1, 28, 28),
        help="Shape of the input data (default: (1, 28, 28))",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=["relu", "tanh", "sigmoid", "elu", "leakyrelu", "gelu", "silu"],
        help="Activation function for encoder and decoder (default: tanh)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for training and testing (default: 100)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Maximum gradient norm for clipping; set to None to disable (default: None)",
    )

    # VAE-specific options
    parser.add_argument(
        "--use-softplus-std",
        action="store_true",
        default=False,
        help="Use softplus parameterization for standard deviation in latent space (default: False)",
    )
    parser.add_argument(
        "--n-latent-samples",
        type=int,
        default=1,
        help="Number of latent samples per input for ELBO estimation (default: 1)",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory for saving runs (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="How often to log training metrics (default: 5 steps)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )

    # Performance and analysis options
    parser.add_argument(
        "--analyze-gradients",
        action="store_true",
        default=False,
        help="Enable detailed gradient analysis (slower but provides gradient diagnostics)",
    )

    # Evaluation and visualization
    parser.add_argument(
        "--max-latent-batches",
        type=int,
        default=400,
        help="Max batches to collect for latent visualization (default: 400)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of samples to generate for visualization (default: 64)",
    )
    parser.add_argument(
        "--n-recon",
        type=int,
        default=16,
        help="Number of reconstructions to visualize (default: 16)",
    )
    parser.add_argument(
        "--interp-steps",
        type=int,
        default=15,
        help="Number of steps for latent interpolation between two datapoints (default: 15)",
    )
    parser.add_argument(
        "--interp-method",
        type=str,
        choices=["slerp", "lerp"],
        default="slerp",
        help="Interpolation method in latent space (default: slerp)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def main():
    """Main training function using VAETrainer."""
    args = parse_args()

    # Create trainer configuration
    trainer_config = TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        run_dir=args.run_dir,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        analyze_gradients=args.analyze_gradients,
        max_latent_batches=args.max_latent_batches,
        n_samples=args.n_samples,
        n_recon=args.n_recon,
        interp_steps=args.interp_steps,
        interp_method=args.interp_method,
        n_latent_samples=args.n_latent_samples,
        device=args.device,
        seed=args.seed,
    )

    # Create model configuration
    model_config = VAEConfig(
        input_shape=args.input_shape,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        activation=args.activation,
        use_softplus_std=args.use_softplus_std,
        n_samples=args.n_latent_samples,
    )

    # Create model
    model = VAE(model_config)
    print(model)

    # Create data loaders
    train_loader, test_loader = get_dataloaders(batch_size=trainer_config.batch_size)

    # Create and run trainer
    trainer = VAETrainer(model=model, config=trainer_config)

    # Save CLI arguments and configurations for tracking
    args_dict = vars(args)
    args_file = os.path.join(trainer.run_dir, "cli_args.json")
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=2, default=str)
    print(f"CLI arguments saved to: {args_file}")
    
    # Save full configuration (trainer + model configs)
    config_dict = {
        "trainer_config": trainer_config.__dict__,
        "model_config": model_config.__dict__,
        "cli_args": args_dict
    }
    config_file = os.path.join(trainer.run_dir, "full_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Full configuration saved to: {config_file}")

    try:
        # Run complete training
        trainer.train(train_loader, test_loader)

        # Save performance metrics
        performance_file = os.path.join(trainer.run_dir, "performance.json")
        trainer.save_performance_metrics(performance_file)
        print(f"Performance metrics saved to: {performance_file}")

        # Load best model and generate analysis plots
        trainer.load_best_model()
        trainer.generate_analysis_plots(test_loader)

    finally:
        # Clean up
        trainer.close()

    print("Training and analysis completed successfully!")


if __name__ == "__main__":
    main()
