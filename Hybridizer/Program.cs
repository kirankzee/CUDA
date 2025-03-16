using System;
using System.Threading.Tasks;
using HybridizerSample.Kernels;
using HybridizerSample.Utils;
using HybridizerSample.Models;

namespace HybridizerSample
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("C# CUDA-like Sample using Hybridizer");
            Console.WriteLine("=====================================");
            Console.WriteLine();

            // Initialize device manager to check for CUDA availability
            var deviceManager = new DeviceManager();
            Console.WriteLine($"CUDA Available: {deviceManager.CudaAvailable}");
            Console.WriteLine($"Device: {deviceManager.DeviceName}");
            Console.WriteLine();

            // Demonstrate vector addition
            DemonstrateVectorAddition();

            // Demonstrate scalar multiplication
            DemonstrateScalarMultiplication();

            // Demonstrate matrix multiplication
            DemonstrateMatrixMultiplication();

            // Demonstrate image convolution
            DemonstrateImageConvolution();

            // Demonstrate multiple streams
            DemonstrateMultipleStreams();

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        static void DemonstrateVectorAddition()
        {
            Console.WriteLine("Vector Addition:");
            Console.WriteLine("-----------------");

            // Initialize data
            int n = 10_000_000;
            float[] a = new float[n];
            float[] b = new float[n];
            float[] c_cpu = new float[n];
            float[] c_gpu = new float[n];

            // Fill arrays with sample data
            Random rand = new Random(42);
            for (int i = 0; i < n; i++)
            {
                a[i] = (float)rand.NextDouble();
                b[i] = (float)rand.NextDouble();
            }

            // Create performance timer
            var timer = new PerformanceTimer("Vector Addition");

            // Run CPU implementation
            Console.WriteLine("Running CPU implementation...");
            timer.Start();
            VectorKernels.VectorAddCPU(a, b, c_cpu, n);
            timer.StopCpu();

            // Run GPU implementation
            Console.WriteLine("Running GPU implementation...");
            timer.Start();
            // In a real implementation with Hybridizer, this would run on the GPU
            VectorKernels.VectorAdd(a, b, c_gpu, n);
            timer.StopGpu();

            // Verify results
            bool correct = VerifyResults(c_cpu, c_gpu, n);
            Console.WriteLine($"Results are correct: {correct}");

            // Print performance results
            timer.PrintResults();
        }

        static void DemonstrateScalarMultiplication()
        {
            Console.WriteLine("Scalar Multiplication:");
            Console.WriteLine("----------------------");

            // Initialize data
            int n = 10_000_000;
            float[] a = new float[n];
            float[] b_cpu = new float[n];
            float[] b_gpu = new float[n];
            float scalar = 3.14f;

            // Fill arrays with sample data
            Random rand = new Random(42);
            for (int i = 0; i < n; i++)
            {
                a[i] = (float)rand.NextDouble();
            }

            // Create performance timer
            var timer = new PerformanceTimer("Scalar Multiplication");

            // Run CPU implementation
            Console.WriteLine("Running CPU implementation...");
            timer.Start();
            VectorKernels.ScalarMultiplyCPU(a, b_cpu, scalar, n);
            timer.StopCpu();

            // Run GPU implementation
            Console.WriteLine("Running GPU implementation...");
            timer.Start();
            // In a real implementation with Hybridizer, this would run on the GPU
            VectorKernels.ScalarMultiply(a, b_gpu, scalar, n);
            timer.StopGpu();

            // Verify results
            bool correct = VerifyResults(b_cpu, b_gpu, n);
            Console.WriteLine($"Results are correct: {correct}");

            // Print performance results
            timer.PrintResults();
        }

        static void DemonstrateMatrixMultiplication()
        {
            Console.WriteLine("Matrix Multiplication:");
            Console.WriteLine("----------------------");

            // Initialize data
            int m = 1000;
            int n = 1000;
            int k = 1000;
            float[] A = new float[m * k];
            float[] B = new float[k * n];
            float[] C_cpu = new float[m * n];
            float[] C_gpu_standard = new float[m * n];
            float[] C_gpu_shared = new float[m * n];

            // Fill matrices with sample data
            Random rand = new Random(42);
            for (int i = 0; i < m * k; i++)
            {
                A[i] = (float)rand.NextDouble();
            }
            for (int i = 0; i < k * n; i++)
            {
                B[i] = (float)rand.NextDouble();
            }

            // Create performance timer
            var timerStandard = new PerformanceTimer("Matrix Multiplication (Standard)");
            var timerShared = new PerformanceTimer("Matrix Multiplication (Shared Memory)");

            // Run CPU implementation
            Console.WriteLine("Running CPU implementation...");
            timerStandard.Start();
            MatrixKernels.MatrixMultiplyCPU(A, B, C_cpu, m, n, k);
            timerStandard.StopCpu();
            timerShared.CpuElapsedMs = timerStandard.CpuElapsedMs;

            // Run GPU standard implementation
            Console.WriteLine("Running GPU standard implementation...");
            timerStandard.Start();
            // In a real implementation with Hybridizer, this would run on the GPU
            MatrixKernels.MatrixMultiply(A, B, C_gpu_standard, m, n, k);
            timerStandard.StopGpu();

            // Run GPU shared memory implementation
            Console.WriteLine("Running GPU shared memory implementation...");
            timerShared.Start();
            // In a real implementation with Hybridizer, this would run on the GPU
            // using shared memory for better performance
            MatrixKernels.MatrixMultiplyShared(A, B, C_gpu_shared, m, n, k);
            timerShared.StopGpu();

            // Verify results
            bool correctStandard = VerifyResults(C_cpu, C_gpu_standard, m * n, 1e-5f);
            bool correctShared = VerifyResults(C_cpu, C_gpu_shared, m * n, 1e-5f);
            Console.WriteLine($"Standard results are correct: {correctStandard}");
            Console.WriteLine($"Shared memory results are correct: {correctShared}");

            // Print performance results
            timerStandard.PrintResults();
            timerShared.PrintResults();
        }

        static void DemonstrateImageConvolution()
        {
            Console.WriteLine("Image Convolution:");
            Console.WriteLine("------------------");

            // Initialize data
            int width = 1024;
            int height = 1024;
            int filterWidth = 5;
            int filterHeight = 5;
            float[] input = new float[width * height];
            float[] output_cpu = new float[width * height];
            float[] output_gpu = new float[width * height];
            float[] filter = new float[filterWidth * filterHeight];

            // Fill input image with sample data
            Random rand = new Random(42);
            for (int i = 0; i < width * height; i++)
            {
                input[i] = (float)rand.NextDouble();
            }

            // Create a Gaussian filter for demonstration
            CreateGaussianFilter(filter, filterWidth, filterHeight, 1.0f);

            // Create performance timer
            var timer = new PerformanceTimer("Image Convolution");

            // Run CPU implementation
            Console.WriteLine("Running CPU implementation...");
            timer.Start();
            ConvolutionKernels.ImageConvolutionCPU(input, output_cpu, filter, width, height, filterWidth, filterHeight);
            timer.StopCpu();

            // Run GPU implementation
            Console.WriteLine("Running GPU implementation...");
            timer.Start();
            // In a real implementation with Hybridizer, this would run on the GPU
            ConvolutionKernels.ImageConvolution(input, output_gpu, filter, width, height, filterWidth, filterHeight);
            timer.StopGpu();

            // Verify results
            bool correct = VerifyResults(output_cpu, output_gpu, width * height, 1e-5f);
            Console.WriteLine($"Results are correct: {correct}");

            // Print performance results
            timer.PrintResults();
        }

        static void DemonstrateMultipleStreams()
        {
            Console.WriteLine("Multiple Streams:");
            Console.WriteLine("-----------------");

            var deviceManager = new DeviceManager();
            if (!deviceManager.CudaAvailable)
            {
                Console.WriteLine("CUDA device not available, skipping multiple streams demonstration");
                Console.WriteLine();
                return;
            }

            // Create multiple streams
            int numStreams = 4;
            Console.WriteLine($"Creating {numStreams} CUDA streams...");
            var streams = deviceManager.CreateStreams(numStreams);

            if (streams != null)
            {
                // In a real implementation, we would distribute work across multiple streams
                // to enable concurrent kernel execution
                Console.WriteLine($"Created {streams.Length} streams successfully");
                Console.WriteLine("In a real implementation, multiple kernels would be launched in different streams");
                Console.WriteLine("to enable concurrent execution and overlap of computation with data transfers.");
            }
            else
            {
                Console.WriteLine("Failed to create streams");
            }
            
            Console.WriteLine();

            // Synchronize all streams
            deviceManager.Synchronize();
        }

        static void CreateGaussianFilter(float[] filter, int width, int height, float sigma)
        {
            // Create a Gaussian filter for image convolution
            float sum = 0.0f;
            int halfWidth = width / 2;
            int halfHeight = height / 2;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float dx = x - halfWidth;
                    float dy = y - halfHeight;
                    float value = (float)Math.Exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                    filter[y * width + x] = value;
                    sum += value;
                }
            }

            // Normalize the filter
            for (int i = 0; i < width * height; i++)
            {
                filter[i] /= sum;
            }
        }

        static bool VerifyResults<T>(T[] expected, T[] actual, int n, float tolerance = 0.0f) where T : IComparable
        {
            if (expected.Length != actual.Length || expected.Length < n)
            {
                return false;
            }

            if (typeof(T) == typeof(float))
            {
                float[] expectedFloat = expected as float[];
                float[] actualFloat = actual as float[];

                for (int i = 0; i < n; i++)
                {
                    if (Math.Abs(expectedFloat[i] - actualFloat[i]) > tolerance)
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!expected[i].Equals(actual[i]))
                    {
                        return false;
                    }
                }
                return true;
            }
        }
    }
}
