## License

This project is licensed under the MIT License.

Using Hybridizer for CUDA-like operations in C# offers several benefits:
1. Ease of Use: Hybridizer allows developers to write GPU-accelerated code using familiar C# syntax, reducing the learning curve associated with CUDA and GPU programming.
2. Performance: By offloading compute-intensive tasks to the GPU, Hybridizer can significantly improve the performance of applications, especially for parallelizable tasks such as matrix multiplication, image processing, and scientific computations.
3. Productivity: Developers can leverage existing C# code and libraries, making it easier to integrate GPU acceleration into existing projects without rewriting large portions of code in CUDA C/C++.
4. Cross-Platform Development: Hybridizer abstracts the complexity of GPU programming, enabling developers to write code that can run on different GPU architectures with minimal changes.
5. Debugging and Profiling: Hybridizer integrates with Visual Studio, providing tools for debugging and profiling GPU code, which helps in identifying performance bottlenecks and ensuring correctness.
6. Seamless Integration: Hybridizer supports seamless integration with .NET applications, allowing developers to take advantage of the rich ecosystem of .NET libraries and tools.
7. Concurrency: Hybridizer supports concurrent execution and data transfers, enabling more efficient use of GPU resources and improving overall application performance.
8. Community and Support: Being part of the .NET ecosystem, developers can benefit from community support, documentation, and resources available for C# and Hybridizer.

By leveraging these benefits, developers can create high-performance applications that take full advantage of modern GPU hardware while maintaining the productivity and ease of use provided by the C# language and .NET framework.
# Hybridizer Sample

This project demonstrates the use of Hybridizer to perform CUDA-like operations in C#.

## Overview

The project includes several demonstrations of GPU-accelerated computations using Hybridizer. The main operations demonstrated are:

1. Vector Addition
2. Scalar Multiplication
3. Matrix Multiplication
4. Image Convolution
5. Multiple Streams

## How to Run

1. Ensure you have the necessary CUDA drivers and Hybridizer installed.
2. Open the solution in Visual Studio.
3. Build and run the project.

## Results


C# CUDA-like Sample using Hybridizer
=====================================

CUDA device found: NVIDIA GeForce RTX Simulator
CUDA Available: True
Device: NVIDIA GeForce RTX Simulator

Vector Addition:
-----------------
Running CPU implementation...
Running GPU implementation...
Results are correct: True
Performance Results for Vector Addition:
  CPU Time: 20 ms
  GPU Time: 178 ms
  Speedup: 0.11x

Scalar Multiplication:
----------------------
Running CPU implementation...
Running GPU implementation...
Results are correct: True
Performance Results for Scalar Multiplication:
  CPU Time: 17 ms
  GPU Time: 41 ms
  Speedup: 0.41x

Matrix Multiplication:
----------------------
Running CPU implementation...
Running GPU standard implementation...
Running GPU shared memory implementation...
Standard results are correct: True
Shared memory results are correct: False
Performance Results for Matrix Multiplication (Standard):
  CPU Time: 1906 ms
  GPU Time: 357 ms
  Speedup: 5.34x

Performance Results for Matrix Multiplication (Shared Memory):
  CPU Time: 1906 ms
  GPU Time: 0 ms
  Speedup: N/A (GPU not available)

Image Convolution:
------------------
Running CPU implementation...
Running GPU implementation...
Results are correct: True
Performance Results for Image Convolution:
  CPU Time: 72 ms
  GPU Time: 10 ms
  Speedup: 7.20x

Multiple Streams:
-----------------
CUDA device found: NVIDIA GeForce RTX Simulator
Creating 4 CUDA streams...
Created 4 streams successfully
In a real implementation, multiple kernels would be launched in different streams
to enable concurrent execution and overlap of computation with data transfers.

Device synchronized