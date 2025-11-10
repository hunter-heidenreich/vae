uv run main.py --analyze-gradients --lr 1e-1 > lr_1e-1.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --lr 1e-2 > lr_1e-2.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --lr 1e-3 > lr_1e-3.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --lr 1e-4 > lr_1e-4.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --lr 1e-5 > lr_1e-5.log 2>&1 &
sleep 2

wait