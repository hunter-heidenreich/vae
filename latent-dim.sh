uv run main.py --analyze-gradients --latent-dim 1 > latent_dim_1.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --latent-dim 2 > latent_dim_2.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --latent-dim 4 > latent_dim_4.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --latent-dim 8 > latent_dim_8.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --latent-dim 16 > latent_dim_16.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --latent-dim 32 > latent_dim_32.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --latent-dim 64 > latent_dim_64.log 2>&1 &
sleep 2

wait