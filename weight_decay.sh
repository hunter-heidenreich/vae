uv run main.py --analyze-gradients --weight-decay 1 > weight_decay_1.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --weight-decay 0.1 > weight_decay_0.1.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --weight-decay 0.01 > weight_decay_0.01.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --weight-decay 0.001 > weight_decay_0.001.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --weight-decay 0.0001 > weight_decay_0.0001.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --weight-decay 0.00001 > weight_decay_0.00001.log 2>&1 &
sleep 2

wait