uv run main.py --analyze-gradients --hidden-dim 32 > hidden_32.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --hidden-dim 64 > hidden_64.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --hidden-dim 128 > hidden_128.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --hidden-dim 256 > hidden_256.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --hidden-dim 512 > hidden_512.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --hidden-dim 1024 > hidden_1024.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --hidden-dim 2048 > hidden_2048.log 2>&1 &
sleep 2

wait