#!/usr/bin/env bash

gcc -Wall -o ImCreate_test_gpuipc ImCreate_test_gpuipc.c ImageStreamIO.c -pthread -DHAVE_CUDA -I/usr/local/cuda/include -std=gnu11 -O0 -g -march=native -fopenmp -flto -pipe -DHAVE_CUDA /usr/local/cuda/lib64/libcudart_static.a -lpthread -ldl -lrt -lcuda
gcc -Wall -o ImCreate_test_gpuipc2 ImCreate_test_gpuipc2.c ImageStreamIO.c -pthread -DHAVE_CUDA -I/usr/local/cuda/include -std=gnu11 -O0 -g -march=native -fopenmp -flto -pipe -DHAVE_CUDA /usr/local/cuda/lib64/libcudart_static.a -lpthread -ldl -lrt
