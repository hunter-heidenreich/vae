uv run main.py --analyze-gradients --batch-size 25 --num-epochs 25 > batch_size_25.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --batch-size 50 --num-epochs 50 > batch_size_50.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --batch-size 100 --num-epochs 100 > batch_size_100.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --batch-size 200 --num-epochs 200 > batch_size_200.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --batch-size 500 --num-epochs 500 > batch_size_500.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --batch-size 1000 --num-epochs 1000 > batch_size_1000.log 2>&1 &
sleep 2

wait