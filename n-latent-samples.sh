uv run main.py --analyze-gradients --n-latent-samples 1 > n_latent_1.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --n-latent-samples 2 > n_latent_2.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --n-latent-samples 4 > n_latent_4.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --n-latent-samples 8 > n_latent_8.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --n-latent-samples 16 > n_latent_16.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --n-latent-samples 32 > n_latent_32.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --n-latent-samples 64 > n_latent_64.log 2>&1 &
sleep 2

wait