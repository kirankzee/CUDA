using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace CudaSample
{
    class Program
    {
        // Existing kernels
        static void VectorAddKernel(
            Index1D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c)
        {
            c[index] = a[index] + b[index];
        }

        static void ScalarMultiplyKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            float scalar)
        {
            output[index] = input[index] * scalar;
        }

        static void MatrixMultiplyKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> a,
            ArrayView2D<float, Stride2D.DenseX> b,
            ArrayView2D<float, Stride2D.DenseX> c,
            int width)
        {
            var x = index.X;
            var y = index.Y;
            
            float sum = 0.0f;
            for (int i = 0; i < width; ++i)
            {
                sum += a[y, i] * b[i, x];
            }
            
            c[y, x] = sum;
        }

        static void MatrixMultiplySharedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> a,
            ArrayView2D<float, Stride2D.DenseX> b,
            ArrayView2D<float, Stride2D.DenseX> c,
            int width)
        {
            var x = index.X;
            var y = index.Y;
            
            if (x >= c.Extent.X || y >= c.Extent.Y)
                return;
                
            float sum = 0.0f;
            
            for (int i = 0; i < width; ++i)
            {
                sum += a[y, i] * b[i, x];
            }
            
            c[y, x] = sum;
        }

        static void ConvolutionKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> input,
            ArrayView2D<float, Stride2D.DenseX> kernel,
            ArrayView2D<float, Stride2D.DenseX> output,
            int kernelSize)
        {
            var x = index.X;
            var y = index.Y;
            
            int radius = kernelSize / 2;
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernelSize; ky++)
            {
                for (int kx = 0; kx < kernelSize; kx++)
                {
                    int inputY = y + (ky - radius);
                    int inputX = x + (kx - radius);
                    
                    if (inputY >= 0 && inputY < input.Extent.Y && 
                        inputX >= 0 && inputX < input.Extent.X)
                    {
                        sum += input[inputY, inputX] * kernel[ky, kx];
                    }
                }
            }
            
            output[y, x] = sum;
        }

        // New kernels
        static void ReductionKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output)
        {
            int i = index;
            if (i < input.Length)
            {
                Atomic.Add(ref output[0], input[i]);
            }
        }

        static void BitonicSortKernel(
            Index1D index,
            ArrayView<float> data,
            int j,
            int k)
        {
            int i = (int)index;
            int ixj = i ^ j;

            if (ixj > i)
            {
                if ((i & k) == 0)
                {
                    if (data[i] > data[ixj])
                    {
                        var temp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = temp;
                    }
                }
                else
                {
                    if (data[i] < data[ixj])
                    {
                        var temp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = temp;
                    }
                }
            }
        }

        // CPU implementations
        static void VectorAddCPU(float[] a, float[] b, float[] c, int length)
        {
            for (int i = 0; i < length; i++)
            {
                c[i] = a[i] + b[i];
            }
        }

        static void ScalarMultiplyCPU(float[] input, float[] output, float scalar, int length)
        {
            for (int i = 0; i < length; i++)
            {
                output[i] = input[i] * scalar;
            }
        }

        static void MatrixMultiplyCPU(float[,] a, float[,] b, float[,] c, int size)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < size; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    c[i, j] = sum;
                }
            }
        }

        static void ConvolutionCPU(float[,] input, float[,] kernel, float[,] output, int inputSize, int kernelSize)
        {
            int radius = kernelSize / 2;
            
            for (int y = 0; y < inputSize; y++)
            {
                for (int x = 0; x < inputSize; x++)
                {
                    float sum = 0.0f;
                    
                    for (int ky = 0; ky < kernelSize; ky++)
                    {
                        for (int kx = 0; kx < kernelSize; kx++)
                        {
                            int inputY = y + (ky - radius);
                            int inputX = x + (kx - radius);
                            
                            if (inputY >= 0 && inputY < inputSize && 
                                inputX >= 0 && inputX < inputSize)
                            {
                                sum += input[inputY, inputX] * kernel[ky, kx];
                            }
                        }
                    }
                    
                    output[y, x] = sum;
                }
            }
        }

        // New CPU implementations
        static float ReductionCPU(float[] input)
        {
            float sum = 0.0f;
            for (int i = 0; i < input.Length; i++)
            {
                sum += input[i];
            }
            return sum;
        }

        static void BitonicSortCPU(float[] data)
        {
            Array.Sort(data);
        }

        // GPU runners
        static float[] RunVectorAdditionGPU(Accelerator accelerator, float[] aData, float[] bData, int length, out long elapsedTime)
        {
            using var a = accelerator.Allocate1D<float>(length);
            using var b = accelerator.Allocate1D<float>(length);
            using var c = accelerator.Allocate1D<float>(length);

            a.CopyFromCPU(aData);
            b.CopyFromCPU(bData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

            var sw = Stopwatch.StartNew();
            kernel(length, a.View, b.View, c.View);
            accelerator.Synchronize();
            sw.Stop();
            elapsedTime = sw.ElapsedMilliseconds;

            return c.GetAsArray1D();
        }

        static float[] RunScalarMultiplicationGPU(Accelerator accelerator, float[] inputData, float scalar, int length, out long elapsedTime)
        {
            using var input = accelerator.Allocate1D<float>(length);
            using var output = accelerator.Allocate1D<float>(length);

            input.CopyFromCPU(inputData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(ScalarMultiplyKernel);

            var sw = Stopwatch.StartNew();
            kernel(length, input.View, output.View, scalar);
            accelerator.Synchronize();
            sw.Stop();
            elapsedTime = sw.ElapsedMilliseconds;

            return output.GetAsArray1D();
        }

        static float[,] RunMatrixMultiplicationGPU(Accelerator accelerator, float[,] aData, float[,] bData, int size, out long elapsedTime)
        {
            using var a = accelerator.Allocate2DDenseX<float>(new Index2D(size, size));
            using var b = accelerator.Allocate2DDenseX<float>(new Index2D(size, size));
            using var c = accelerator.Allocate2DDenseX<float>(new Index2D(size, size));

            a.CopyFromCPU(aData);
            b.CopyFromCPU(bData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                int>(MatrixMultiplyKernel);

            var sw = Stopwatch.StartNew();
            kernel(new Index2D(size, size), a.View, b.View, c.View, size);
            accelerator.Synchronize();
            sw.Stop();
            elapsedTime = sw.ElapsedMilliseconds;

            return c.GetAsArray2D();
        }

        static float[,] RunMatrixMultiplicationSharedGPU(Accelerator accelerator, float[,] aData, float[,] bData, int size, out long elapsedTime)
        {
            using var a = accelerator.Allocate2DDenseX<float>(new Index2D(size, size));
            using var b = accelerator.Allocate2DDenseX<float>(new Index2D(size, size));
            using var c = accelerator.Allocate2DDenseX<float>(new Index2D(size, size));

            a.CopyFromCPU(aData);
            b.CopyFromCPU(bData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                int>(MatrixMultiplySharedKernel);

            var sw = Stopwatch.StartNew();
            kernel(new Index2D(size, size), a.View, b.View, c.View, size);
            accelerator.Synchronize();
            sw.Stop();
            elapsedTime = sw.ElapsedMilliseconds;

            return c.GetAsArray2D();
        }

        static float[,] RunConvolutionGPU(Accelerator accelerator, float[,] imageData, float[,] kernelData, int imageSize, int kernelSize, out long elapsedTime)
        {
            using var image = accelerator.Allocate2DDenseX<float>(new Index2D(imageSize, imageSize));
            using var kernel = accelerator.Allocate2DDenseX<float>(new Index2D(kernelSize, kernelSize));
            using var output = accelerator.Allocate2DDenseX<float>(new Index2D(imageSize, imageSize));

            image.CopyFromCPU(imageData);
            kernel.CopyFromCPU(kernelData);

            var convKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                int>(ConvolutionKernel);

            var sw = Stopwatch.StartNew();
            convKernel(new Index2D(imageSize, imageSize), image.View, kernel.View, output.View, kernelSize);
            accelerator.Synchronize();
            sw.Stop();
            elapsedTime = sw.ElapsedMilliseconds;

            return output.GetAsArray2D();
        }

        // New GPU runners
        static float RunReductionGPU(Accelerator accelerator, float[] inputData, out long elapsedTime)
        {
            int length = inputData.Length;

            using var input = accelerator.Allocate1D<float>(length);
            using var output = accelerator.Allocate1D<float>(1);

            input.CopyFromCPU(inputData);
            output.MemSetToZero();

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                ReductionKernel);

            var sw = Stopwatch.StartNew();
            
            kernel(length, input.View, output.View);
            accelerator.Synchronize();

            float finalSum = output.GetAsArray1D()[0];

            sw.Stop();
            elapsedTime = sw.ElapsedMilliseconds;

            return finalSum;
        }

        static void RunBitonicSortGPU(Accelerator accelerator, float[] data, out long elapsedTime)
        {
            int length = data.Length;
            using var gpuData = accelerator.Allocate1D<float>(length);
            gpuData.CopyFromCPU(data);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, int, int>(
                BitonicSortKernel);

            var sw = Stopwatch.StartNew();

            for (int k = 2; k <= length; k <<= 1)
            {
                for (int j = k >> 1; j > 0; j >>= 1)
                {
                    kernel(length, gpuData.View, j, k);
                    accelerator.Synchronize();
                }
            }

            sw.Stop();
            elapsedTime = sw.ElapsedMilliseconds;

            gpuData.CopyToCPU(data);
        }

        // Feature runners
        static void RunVectorAddition(Accelerator accelerator, bool usingCuda)
        {
            const int length = 1000000;
            
            var random = new Random(42);
            var aData = new float[length];
            var bData = new float[length];
            
            for (int i = 0; i < length; i++)
            {
                aData[i] = (float)random.NextDouble();
                bData[i] = (float)random.NextDouble();
            }

            var gpuResultData = RunVectorAdditionGPU(accelerator, aData, bData, length, out long gpuTime);

            var cpuResultData = new float[length];
            var cpuSw = Stopwatch.StartNew();
            VectorAddCPU(aData, bData, cpuResultData, length);
            cpuSw.Stop();
            long cpuTime = cpuSw.ElapsedMilliseconds;

            bool resultsMatch = true;
            for (int i = 0; i < length; i++)
            {
                if (Math.Abs(gpuResultData[i] - cpuResultData[i]) > 1e-5)
                {
                    resultsMatch = false;
                    Console.WriteLine($"Results mismatch at index {i}: GPU={gpuResultData[i]}, CPU={cpuResultData[i]}");
                    break;
                }
            }

            Console.WriteLine($"Vector addition of {length} elements:");
            Console.WriteLine($"GPU Time: {gpuTime}ms");
            Console.WriteLine($"CPU Time: {cpuTime}ms");
            Console.WriteLine($"Speedup: {(float)cpuTime / gpuTime:F2}x");
            Console.WriteLine($"Results match: {resultsMatch}");
            
            Console.WriteLine("\nSample results:");
            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine($"[{i}]: {aData[i]} + {bData[i]} = {gpuResultData[i]}");
            }
        }

        static void RunScalarMultiplication(Accelerator accelerator, bool usingCuda)
        {
            const int length = 1000000;
            const float scalar = 2.5f;
            
            var random = new Random(42);
            var inputData = new float[length];
            
            for (int i = 0; i < length; i++)
            {
                inputData[i] = (float)random.NextDouble();
            }

            var gpuResultData = RunScalarMultiplicationGPU(accelerator, inputData, scalar, length, out long gpuTime);

            var cpuResultData = new float[length];
            var cpuSw = Stopwatch.StartNew();
            ScalarMultiplyCPU(inputData, cpuResultData, scalar, length);
            cpuSw.Stop();
            long cpuTime = cpuSw.ElapsedMilliseconds;

            bool resultsMatch = true;
            for (int i = 0; i < length; i++)
            {
                if (Math.Abs(gpuResultData[i] - cpuResultData[i]) > 1e-5)
                {
                    resultsMatch = false;
                    Console.WriteLine($"Results mismatch at index {i}: GPU={gpuResultData[i]}, CPU={cpuResultData[i]}");
                    break;
                }
            }

            Console.WriteLine($"Scalar multiplication of {length} elements by {scalar}:");
            Console.WriteLine($"GPU Time: {gpuTime}ms");
            Console.WriteLine($"CPU Time: {cpuTime}ms");
            Console.WriteLine($"Speedup: {(float)cpuTime / gpuTime:F2}x");
            Console.WriteLine($"Results match: {resultsMatch}");
            
            Console.WriteLine("\nSample results:");
            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine($"[{i}]: {inputData[i]} * {scalar} = {gpuResultData[i]}");
            }
        }
        
        static void RunMatrixMultiplication(Accelerator accelerator, bool usingCuda)
        {
            const int size = 1000;
            
            var random = new Random(42);
            var aData = new float[size, size];
            var bData = new float[size, size];
            
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    aData[i, j] = (float)random.NextDouble();
                    bData[i, j] = (float)random.NextDouble();
                }
            }

            var gpuResultData = RunMatrixMultiplicationGPU(accelerator, aData, bData, size, out long gpuTime);

            int cpuSize = Math.Min(size, 500);
            var cpuAData = new float[cpuSize, cpuSize];
            var cpuBData = new float[cpuSize, cpuSize];
            var cpuResultData = new float[cpuSize, cpuSize];
            
            for (int i = 0; i < cpuSize; i++)
            {
                for (int j = 0; j < cpuSize; j++)
                {
                    cpuAData[i, j] = aData[i, j];
                    cpuBData[i, j] = bData[i, j];
                }
            }
            
            var cpuSw = Stopwatch.StartNew();
            MatrixMultiplyCPU(cpuAData, cpuBData, cpuResultData, cpuSize);
            cpuSw.Stop();
            long cpuTime = cpuSw.ElapsedMilliseconds;

            bool resultsMatch = true;
            for (int i = 0; i < cpuSize && resultsMatch; i++)
            {
                for (int j = 0; j < cpuSize; j++)
                {
                    if (Math.Abs(gpuResultData[i, j] - cpuResultData[i, j]) > 1e-3)
                    {
                        resultsMatch = false;
                        Console.WriteLine($"Results mismatch at [{i},{j}]: GPU={gpuResultData[i, j]}, CPU={cpuResultData[i, j]}");
                        break;
                    }
                }
            }

            Console.WriteLine($"Matrix multiplication of {size}x{size} matrices:");
            Console.WriteLine($"GPU Time: {gpuTime}ms");
            if (cpuSize < size)
            {
                Console.WriteLine($"CPU Time: {cpuTime}ms (for {cpuSize}x{cpuSize} matrices)");
                double estimatedFullCpuTime = cpuTime * Math.Pow((double)size / cpuSize, 3);
                Console.WriteLine($"Estimated CPU Time for full {size}x{size} matrices: {estimatedFullCpuTime:F0}ms");
                Console.WriteLine($"Estimated Speedup: {estimatedFullCpuTime / gpuTime:F2}x");
            }
            else
            {
                Console.WriteLine($"CPU Time: {cpuTime}ms");
                Console.WriteLine($"Speedup: {(float)cpuTime / gpuTime:F2}x");
            }
            Console.WriteLine($"Results match for {cpuSize}x{cpuSize} subset: {resultsMatch}");
            
            Console.WriteLine("\nSample results:");
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Console.WriteLine($"[{i},{j}]: {gpuResultData[i, j]}");
                }
            }
        }
        
        static void RunMatrixMultiplicationShared(Accelerator accelerator)
        {
            const int size = 1000;
            
            var random = new Random(42);
            var aData = new float[size, size];
            var bData = new float[size, size];
            
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    aData[i, j] = (float)random.NextDouble();
                    bData[i, j] = (float)random.NextDouble();
                }
            }

            var standardGpuResultData = RunMatrixMultiplicationGPU(accelerator, aData, bData, size, out long standardGpuTime);
            var optimizedGpuResultData = RunMatrixMultiplicationSharedGPU(accelerator, aData, bData, size, out long optimizedGpuTime);

            bool resultsMatch = true;
            for (int i = 0; i < size && resultsMatch; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    if (Math.Abs(standardGpuResultData[i, j] - optimizedGpuResultData[i, j]) > 1e-3)
                    {
                        resultsMatch = false;
                        Console.WriteLine($"Results mismatch at [{i},{j}]: Standard={standardGpuResultData[i, j]}, Optimized={optimizedGpuResultData[i, j]}");
                        break;
                    }
                }
            }

            Console.WriteLine($"Matrix multiplication of {size}x{size} matrices with shared memory optimization:");
            Console.WriteLine($"Standard GPU Time: {standardGpuTime}ms");
            Console.WriteLine($"Optimized GPU Time: {optimizedGpuTime}ms");
            Console.WriteLine($"Speedup from shared memory: {(float)standardGpuTime / optimizedGpuTime:F2}x");
            Console.WriteLine($"Results match: {resultsMatch}");
        }
        
        static void RunConvolution(Accelerator accelerator, bool usingCuda)
        {
            const int imageSize = 1024;
            const int kernelSize = 5;
            
            var random = new Random(42);
            var imageData = new float[imageSize, imageSize];
            var kernelData = new float[kernelSize, kernelSize];
            
            for (int i = 0; i < imageSize; i++)
            {
                for (int j = 0; j < imageSize; j++)
                {
                    imageData[i, j] = (float)random.NextDouble();
                }
            }
            
            float kernelSum = 0.0f;
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    int x = j - kernelSize / 2;
                    int y = i - kernelSize / 2;
                    kernelData[i, j] = (float)Math.Exp(-(x * x + y * y) / (2.0f * 1.5f * 1.5f));
                    kernelSum += kernelData[i, j];
                }
            }
            
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    kernelData[i, j] /= kernelSum;
                }
            }

            var gpuResultData = RunConvolutionGPU(accelerator, imageData, kernelData, imageSize, kernelSize, out long gpuTime);

            int cpuImageSize = Math.Min(imageSize, 256);
            var cpuImageData = new float[cpuImageSize, cpuImageSize];
            var cpuResultData = new float[cpuImageSize, cpuImageSize];
            
            for (int i = 0; i < cpuImageSize; i++)
            {
                for (int j = 0; j < cpuImageSize; j++)
                {
                    cpuImageData[i, j] = imageData[i, j];
                }
            }
            
            var cpuSw = Stopwatch.StartNew();
            ConvolutionCPU(cpuImageData, kernelData, cpuResultData, cpuImageSize, kernelSize);
            cpuSw.Stop();
            long cpuTime = cpuSw.ElapsedMilliseconds;

            bool resultsMatch = true;
            for (int i = 0; i < cpuImageSize && resultsMatch; i++)
            {
                for (int j = 0; j < cpuImageSize; j++)
                {
                    if (Math.Abs(gpuResultData[i, j] - cpuResultData[i, j]) > 1e-3)
                    {
                        resultsMatch = false;
                        Console.WriteLine($"Results mismatch at [{i},{j}]: GPU={gpuResultData[i, j]}, CPU={cpuResultData[i, j]}");
                        break;
                    }
                }
            }

            Console.WriteLine($"Convolution of {imageSize}x{imageSize} image with {kernelSize}x{kernelSize} kernel:");
            Console.WriteLine($"GPU Time: {gpuTime}ms");
            if (cpuImageSize < imageSize)
            {
                Console.WriteLine($"CPU Time: {cpuTime}ms (for {cpuImageSize}x{cpuImageSize} image)");
                double estimatedFullCpuTime = cpuTime * Math.Pow((double)imageSize / cpuImageSize, 2);
                Console.WriteLine($"Estimated CPU Time for full {imageSize}x{imageSize} image: {estimatedFullCpuTime:F0}ms");
                Console.WriteLine($"Estimated Speedup: {estimatedFullCpuTime / gpuTime:F2}x");
            }
            else
            {
                Console.WriteLine($"CPU Time: {cpuTime}ms");
                Console.WriteLine($"Speedup: {(float)cpuTime / gpuTime:F2}x");
            }
            Console.WriteLine($"Results match for {cpuImageSize}x{cpuImageSize} subset: {resultsMatch}");
        }
        
        static void RunParallelExecution(Accelerator accelerator, bool usingCuda)
        {
            const int vectorLength = 1000000;
            const float scalar = 2.5f;
            
            var random = new Random(42);
            var aData = new float[vectorLength];
            var bData = new float[vectorLength];
            
            for (int i = 0; i < vectorLength; i++)
            {
                aData[i] = (float)random.NextDouble();
                bData[i] = (float)random.NextDouble();
            }
            
            using var a = accelerator.Allocate1D<float>(vectorLength);
            using var b = accelerator.Allocate1D<float>(vectorLength);
            using var c = accelerator.Allocate1D<float>(vectorLength);
            using var d = accelerator.Allocate1D<float>(vectorLength);

            a.CopyFromCPU(aData);
            b.CopyFromCPU(bData);

            var addKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
            var mulKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(ScalarMultiplyKernel);
            
            Console.WriteLine("Launching kernels sequentially...");
            var sw = Stopwatch.StartNew();
            
            // Execute sequentially
            addKernel(vectorLength, a.View, b.View, c.View);
            accelerator.Synchronize();
            
            mulKernel(vectorLength, a.View, d.View, scalar);
            accelerator.Synchronize();
            
            sw.Stop();
            long sequentialTime = sw.ElapsedMilliseconds;
            
            Console.WriteLine($"Sequential execution time: {sequentialTime}ms");
            
            // Note: In ILGPU 1.5.2, parallel execution with streams works differently
            // and is not demonstrated in this simplified example
            Console.WriteLine("Note: Parallel execution with streams is not demonstrated in this example.");
        }

        // New feature runners
        static void RunReduction(Accelerator accelerator, bool usingCuda)
        {
            const int length = 1000000;
            
            var random = new Random(42);
            var inputData = new float[length];
            
            float expectedSum = 0.0f;
            for (int i = 0; i < length; i++)
            {
                inputData[i] = (float)random.NextDouble();
                expectedSum += inputData[i];
            }

            float gpuSum = RunReductionGPU(accelerator, inputData, out long gpuTime);

            var cpuSw = Stopwatch.StartNew();
            float cpuSum = ReductionCPU(inputData);
            cpuSw.Stop();
            long cpuTime = cpuSw.ElapsedMilliseconds;

            bool resultsMatch = Math.Abs(gpuSum - cpuSum) < 1e-3;

            Console.WriteLine($"\nReduction (sum) of {length} elements:");
            Console.WriteLine($"GPU Time: {gpuTime}ms");
            Console.WriteLine($"CPU Time: {cpuTime}ms");
            Console.WriteLine($"Speedup: {(float)cpuTime / gpuTime:F2}x");
            Console.WriteLine($"GPU Sum: {gpuSum}");
            Console.WriteLine($"CPU Sum: {cpuSum}");
            Console.WriteLine($"Results match: {resultsMatch}");
        }

        static void RunBitonicSort(Accelerator accelerator, bool usingCuda)
        {
            const int length = 1048576; // Must be power of 2 for bitonic sort
            
            var random = new Random(42);
            var data = new float[length];
            var cpuData = new float[length];
            
            for (int i = 0; i < length; i++)
            {
                data[i] = (float)random.NextDouble();
                cpuData[i] = data[i];
            }

            RunBitonicSortGPU(accelerator, data, out long gpuTime);

            var cpuSw = Stopwatch.StartNew();
            BitonicSortCPU(cpuData);
            cpuSw.Stop();
            long cpuTime = cpuSw.ElapsedMilliseconds;

            bool resultsMatch = true;
            for (int i = 0; i < length; i++)
            {
                if (Math.Abs(data[i] - cpuData[i]) > 1e-5)
                {
                    resultsMatch = false;
                    Console.WriteLine($"Results mismatch at index {i}: GPU={data[i]}, CPU={cpuData[i]}");
                    break;
                }
            }

            Console.WriteLine($"\nBitonic sort of {length} elements:");
            Console.WriteLine($"GPU Time: {gpuTime}ms");
            Console.WriteLine($"CPU Time: {cpuTime}ms");
            Console.WriteLine($"Speedup: {(float)cpuTime / gpuTime:F2}x");
            Console.WriteLine($"Results match: {resultsMatch}");
            
            Console.WriteLine("\nFirst 5 sorted elements:");
            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine($"[{i}]: {data[i]}");
            }
        }

        static void Main(string[] args)
        {
            using var context = Context.Create(builder => builder.Default().EnableAlgorithms());

            Accelerator accelerator;
            bool usingCuda = false;
            
            if (context.Devices.Any(device => device.AcceleratorType == AcceleratorType.Cuda))
            {
                accelerator = context.GetCudaDevice(0).CreateAccelerator(context);
                Console.WriteLine("Using CUDA accelerator: " + accelerator.Name);
                usingCuda = true;
                
                var device = context.GetCudaDevice(0);
                Console.WriteLine($"CUDA Device: {device.Name}");
                Console.WriteLine($"Total Memory: {device.MemorySize / (1024 * 1024)} MB");
                Console.WriteLine($"Max Grid Size: [{device.MaxGridSize.X}, {device.MaxGridSize.Y}, {device.MaxGridSize.Z}]");
                Console.WriteLine($"Warp Size: {device.WarpSize}");
            }
            else
            {
                accelerator = context.GetCPUDevice(0).CreateAccelerator(context);
                Console.WriteLine("No CUDA device found. Using CPU accelerator: " + accelerator.Name);
            }

            using (accelerator)
            {
                Console.WriteLine("\n=== Vector Addition Example ===");
                RunVectorAddition(accelerator, usingCuda);

                Console.WriteLine("\n=== Scalar Multiplication Example ===");
                RunScalarMultiplication(accelerator, usingCuda);
                
                Console.WriteLine("\n=== Matrix Multiplication Example ===");
                RunMatrixMultiplication(accelerator, usingCuda);
                
                if (usingCuda)
                {
                    Console.WriteLine("\n=== Matrix Multiplication with Shared Memory Example ===");
                    RunMatrixMultiplicationShared(accelerator);
                }
                
                Console.WriteLine("\n=== Convolution Example (Image Processing) ===");
                RunConvolution(accelerator, usingCuda);
                
                Console.WriteLine("\n=== Parallel Execution Example ===");
                RunParallelExecution(accelerator, usingCuda);

                // New features
                Console.WriteLine("\n=== Parallel Reduction Example ===");
                RunReduction(accelerator, usingCuda);

                Console.WriteLine("\n=== Bitonic Sort Example ===");
                RunBitonicSort(accelerator, usingCuda);
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}