# C# CUDA-like Sample using ILGPU

This project demonstrates GPU computing in C# using ILGPU, which is a high-level library that provides CUDA-like functionality for .NET applications.

## Features

- Vector addition on GPU
- Scalar multiplication on GPU
- Matrix multiplication on GPU (standard and shared memory optimized)
- Image convolution (used in image processing and neural networks)
- Parallel kernel execution using multiple streams
- Automatic fallback to CPU if no CUDA device is available
- Performance comparison between CPU and GPU implementations

## Requirements

- .NET 7.0 or higher
- For GPU acceleration: NVIDIA GPU with CUDA support

## How It Works

The sample demonstrates several common GPU operations:

1. **Vector Addition**: Adding two arrays element by element
2. **Scalar Multiplication**: Multiplying each element of an array by a constant value
3. **Matrix Multiplication**: Multiplying two matrices together (demonstrates the massive parallel processing advantage of GPUs)
4. **Optimized Matrix Multiplication**: Using shared memory to improve performance
5. **Image Convolution**: Applying a filter kernel to an image (common in image processing)
6. **Parallel Execution**: Running multiple kernels simultaneously using CUDA streams

The code automatically detects if a CUDA-capable GPU is available and falls back to CPU execution if none is found.

## Advanced Features

### Shared Memory Optimization

The project includes an optimized matrix multiplication implementation that uses shared memory to reduce global memory access, which is a common optimization technique in CUDA programming.

### Convolution

The convolution kernel demonstrates how to implement image processing operations on the GPU, which is useful for applications like image filtering, edge detection, and neural networks.

### Parallel Execution with Streams

The sample shows how to use multiple CUDA streams to execute kernels in parallel, which can improve overall throughput when running multiple independent operations.

## Performance Comparison

The application includes both GPU and CPU implementations of the same operations and measures:
- GPU execution time
- CPU execution time
- Speedup factor (CPU time / GPU time)

For operations with large data sets, the CPU implementation is limited to a smaller subset to keep execution times reasonable, and the full CPU time is estimated based on algorithmic complexity.

This allows you to directly observe the performance benefits of GPU acceleration compared to traditional CPU processing.

## Notes

- ILGPU supports not only CUDA but also OpenCL and CPU acceleration
- The sample uses 1 million elements for vector operations and 1000x1000 matrices for matrix multiplication
- Matrix multiplication demonstrates the most significant performance advantage due to its O(n³) complexity
- Results may vary significantly based on your hardware configuration