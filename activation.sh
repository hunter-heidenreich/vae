#!/bin/bash

# Activation function sweep script
# Tests different activation functions for the VAE

uv run main.py --analyze-gradients --activation relu > activation_relu.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --activation tanh > activation_tanh.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --activation sigmoid > activation_sigmoid.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --activation elu > activation_elu.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --activation leakyrelu > activation_leakyrelu.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --activation gelu > activation_gelu.log 2>&1 &
sleep 2

uv run main.py --analyze-gradients --activation silu > activation_silu.log 2>&1 &
sleep 2

wait

echo "Activation function sweep completed!"
echo "Results saved to activation_*.log files"