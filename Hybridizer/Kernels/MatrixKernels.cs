using System;
using System.Threading.Tasks;
using HybridizerSample.Models;

namespace HybridizerSample.Kernels
{
    /// <summary>
    /// Matrix operation kernels for GPU execution
    /// </summary>
    public class MatrixKernels
    {
        /// <summary>
        /// Standard matrix multiplication: C = A * B
        /// </summary>
        /// <param name="A">First input matrix</param>
        /// <param name="B">Second input matrix</param>
        /// <param name="C">Output matrix</param>
        /// <param name="m">Rows in A</param>
        /// <param name="n">Columns in B</param>
        /// <param name="k">Columns in A / Rows in B</param>
        [EntryPoint]
        public static void MatrixMultiply(float[] A, float[] B, float[] C, int m, int n, int k)
        {
            Parallel.For(0, m, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0.0f;
                    for (int l = 0; l < k; l++)
                    {
                        sum += A[i * k + l] * B[l * n + j];
                    }
                    C[i * n + j] = sum;
                }
            });
        }

        /// <summary>
        /// CPU fallback implementation of matrix multiplication
        /// </summary>
        public static void MatrixMultiplyCPU(float[] A, float[] B, float[] C, int m, int n, int k)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0.0f;
                    for (int l = 0; l < k; l++)
                    {
                        sum += A[i * k + l] * B[l * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
        }

        /// <summary>
        /// Matrix multiplication using shared memory for better performance
        /// </summary>
        /// <param name="A">First input matrix</param>
        /// <param name="B">Second input matrix</param>
        /// <param name="C">Output matrix</param>
        /// <param name="m">Rows in A</param>
        /// <param name="n">Columns in B</param>
        /// <param name="k">Columns in A / Rows in B</param>
        [EntryPoint]
        [HybridizeLaunchConfig(block: new int[] { 32, 32, 1 })]
        public static void MatrixMultiplyShared(float[] A, float[] B, float[] C, int m, int n, int k)
        {
            // In real implementation with Hybridizer, this would use __shared__ memory
            // to optimize matrix multiplication with tiled algorithm
            // For demonstration purposes, we're using a simplified implementation
            int bx = threadIdx.x;
            int by = threadIdx.y;
            int tx = blockIdx.x * blockDim.x + threadIdx.x;
            int ty = blockIdx.y * blockDim.y + threadIdx.y;

            if (tx < m && ty < n)
            {
                float sum = 0.0f;
                for (int l = 0; l < k; l++)
                {
                    sum += A[tx * k + l] * B[l * n + ty];
                }
                C[tx * n + ty] = sum;
            }
        }
    }
} 