using System;
using System.Threading.Tasks;
using HybridizerSample.Models;

namespace HybridizerSample.Kernels
{
    /// <summary>
    /// Vector operation kernels for GPU execution
    /// </summary>
    public class VectorKernels
    {
        /// <summary>
        /// Adds two vectors element by element: c[i] = a[i] + b[i]
        /// </summary>
        /// <param name="a">First input vector</param>
        /// <param name="b">Second input vector</param>
        /// <param name="c">Output vector</param>
        /// <param name="n">Vector length</param>
        [EntryPoint]
        public static void VectorAdd(float[] a, float[] b, float[] c, int n)
        {
            Parallel.For(0, n, i => { c[i] = a[i] + b[i]; });
        }

        /// <summary>
        /// CPU fallback implementation of vector addition
        /// </summary>
        public static void VectorAddCPU(float[] a, float[] b, float[] c, int n)
        {
            for (int i = 0; i < n; i++)
            {
                c[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// Multiplies a vector by a scalar: b[i] = a[i] * scalar
        /// </summary>
        /// <param name="a">Input vector</param>
        /// <param name="b">Output vector</param>
        /// <param name="scalar">Scalar value</param>
        /// <param name="n">Vector length</param>
        [EntryPoint]
        public static void ScalarMultiply(float[] a, float[] b, float scalar, int n)
        {
            Parallel.For(0, n, i => { b[i] = a[i] * scalar; });
        }

        /// <summary>
        /// CPU fallback implementation of scalar multiplication
        /// </summary>
        public static void ScalarMultiplyCPU(float[] a, float[] b, float scalar, int n)
        {
            for (int i = 0; i < n; i++)
            {
                b[i] = a[i] * scalar;
            }
        }
    }
} 