import argparse
import os
import torch

from src.config import SpCauchyVAEConfig
from src.model import SpCauchyVAE
from src.trainer import Trainer, build_image_dataloaders
from src.utils import plot_samples, plot_reconstructions, set_all_seeds

DATASET_DIMS = {
    "mnist": [1, 28, 28],
    "fashion-mnist": [1, 28, 28],
    "cifar10": [3, 32, 32],
}


def infer_input_dim(dataset):
    if dataset not in DATASET_DIMS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DATASET_DIMS[dataset]


def resolve_device(requested):
    return "cuda" if torch.cuda.is_available() and requested == "cuda" else "cpu"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model(checkpoint_path, device):
    model = SpCauchyVAE.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def run_train(args):
    device = resolve_device(args.device)
    config = SpCauchyVAEConfig(
        input_dim=infer_input_dim(args.dataset),
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        distribution_type=args.distribution,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        dropout_rate=args.dropout_rate,
        kl_weight=args.kl_weight,
        activation=args.activation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    set_all_seeds(config.seed)
    model = SpCauchyVAE(config)
    trainer = Trainer(
        model=model,
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )
    trainer.train()


def run_generate(args):
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    ensure_dir(args.output_dir)
    fig = plot_samples(model, num_samples=args.num_samples, device=device)
    output_path = os.path.join(args.output_dir, "samples.png")
    fig.savefig(output_path)
    print(f"Saved samples to {output_path}")


def run_reconstruct(args):
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    _, val_loader = build_image_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle_train=False,
    )
    ensure_dir(args.output_dir)
    fig = plot_reconstructions(model, val_loader, num_examples=args.num_examples, device=device)
    output_path = os.path.join(args.output_dir, "reconstructions.png")
    fig.savefig(output_path)
    print(f"Saved reconstructions to {output_path}")


def add_model_args(parser):
    parser.add_argument("--latent-dim", type=int, default=3, help="Latent dimension (MNIST tutorial default).")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64, 128], help="Hidden layer sizes (MNIST tutorial default).")
    parser.add_argument("--distribution", choices=["spcauchy", "normal"], default="spcauchy", help="Latent distribution.")
    parser.add_argument("--encoder-type", choices=["mlp", "cnn"], default="cnn", help="Encoder backbone.")
    parser.add_argument("--decoder-type", choices=["mlp", "cnn"], default="cnn", help="Decoder backbone.")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate for MLP/Transformer blocks (0.1 in tutorial).")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="Weight on KL term.")
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "tanh"], default="relu", help="Activation used in MLPs.")
    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(description="spCauchy-VAE CLI helper")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train
    train_p = subparsers.add_parser("train", help="Train a model on an image dataset.")
    add_model_args(train_p)
    train_p.add_argument("--dataset", choices=list(DATASET_DIMS.keys()), default="mnist", help="Dataset to use.")
    train_p.add_argument("--data-dir", default="./data", help="Where to download/read the dataset.")
    train_p.add_argument("--checkpoint-dir", default="./checkpoints", help="Where to store checkpoints.")
    train_p.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs.")
    train_p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    train_p.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    train_p.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Compute device preference.")
    train_p.set_defaults(func=run_train)

    # Generate
    gen_p = subparsers.add_parser("generate", help="Generate samples from a saved checkpoint.")
    gen_p.add_argument("--checkpoint", required=True, help="Path to a saved model checkpoint.")
    gen_p.add_argument("--num-samples", type=int, default=16, help="Number of samples to draw.")
    gen_p.add_argument("--output-dir", default="./outputs", help="Where to save sample grids.")
    gen_p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Compute device preference.")
    gen_p.set_defaults(func=run_generate)

    # Reconstruct
    rec_p = subparsers.add_parser("reconstruct", help="Reconstruct samples from a dataset.")
    rec_p.add_argument("--checkpoint", required=True, help="Path to a saved model checkpoint.")
    rec_p.add_argument("--dataset", choices=list(DATASET_DIMS.keys()), default="mnist", help="Dataset to reconstruct from.")
    rec_p.add_argument("--data-dir", default="./data", help="Where to download/read the dataset.")
    rec_p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for reconstruction loader.")
    rec_p.add_argument("--num-examples", type=int, default=8, help="How many examples to plot.")
    rec_p.add_argument("--output-dir", default="./outputs", help="Where to save reconstruction grids.")
    rec_p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Compute device preference.")
    rec_p.set_defaults(func=run_reconstruct)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
