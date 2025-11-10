uv run main.py --analyze-gradients --seed 42 > seed_42.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --seed 123 > seed_123.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --seed 999 > seed_999.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --seed 2024 > seed_2024.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --seed 0 > seed_0.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --seed 7 > seed_7.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --seed 31415 > seed_31415.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --seed 27182 > seed_27182.log 2>&1 &
sleep 2

wait