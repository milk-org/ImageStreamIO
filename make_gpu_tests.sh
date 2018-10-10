#!env bash

gcc -Wall -o ImCreate_test_gpuipc ImCreate_test_gpuipc.c -L/home/sevin/workspace/greenflash/local/lib -lImageStreamIO -pthread -DHAVE_CUDA -I/opt/cuda/include -std=gnu11 -O0 -g -march=native -fopenmp -flto -pipe -DHAVE_CUDA /opt/cuda/lib64/libcudart_static.a -lpthread -ldl -lrt -lcuda 
gcc -Wall -o ImCreate_test_gpuipc2 ImCreate_test_gpuipc2.c -L/home/sevin/workspace/greenflash/local/lib -lImageStreamIO -pthread -DHAVE_CUDA -I/opt/cuda/include -std=gnu11 -O0 -g -march=native -fopenmp -flto -pipe -DHAVE_CUDA /opt/cuda/lib64/libcudart_static.a -lpthread -ldl -lrt 
