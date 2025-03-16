using System;
using System.Threading.Tasks;
using HybridizerSample.Models;

namespace HybridizerSample.Kernels
{
    /// <summary>
    /// Image convolution kernels for GPU execution
    /// </summary>
    public class ConvolutionKernels
    {
        /// <summary>
        /// Applies a 2D convolution filter to an image
        /// </summary>
        /// <param name="input">Input image data (flattened 2D array)</param>
        /// <param name="output">Output image data (flattened 2D array)</param>
        /// <param name="filter">Convolution filter/kernel (flattened 2D array)</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <param name="filterWidth">Filter width</param>
        /// <param name="filterHeight">Filter height</param>
        [EntryPoint]
        public static void ImageConvolution(float[] input, float[] output, float[] filter,
                                           int width, int height, int filterWidth, int filterHeight)
        {
            // Calculate filter radius
            int filterRadiusX = filterWidth / 2;
            int filterRadiusY = filterHeight / 2;

            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    float sum = 0.0f;

                    // Apply the filter
                    for (int fy = 0; fy < filterHeight; fy++)
                    {
                        int inputY = y + fy - filterRadiusY;
                        
                        // Handle boundary conditions (clamp to edge)
                        if (inputY < 0) inputY = 0;
                        if (inputY >= height) inputY = height - 1;

                        for (int fx = 0; fx < filterWidth; fx++)
                        {
                            int inputX = x + fx - filterRadiusX;
                            
                            // Handle boundary conditions (clamp to edge)
                            if (inputX < 0) inputX = 0;
                            if (inputX >= width) inputX = width - 1;

                            float filterValue = filter[fy * filterWidth + fx];
                            float inputValue = input[inputY * width + inputX];
                            sum += filterValue * inputValue;
                        }
                    }

                    output[y * width + x] = sum;
                }
            });
        }

        /// <summary>
        /// CPU fallback implementation of image convolution
        /// </summary>
        public static void ImageConvolutionCPU(float[] input, float[] output, float[] filter,
                                              int width, int height, int filterWidth, int filterHeight)
        {
            // Calculate filter radius
            int filterRadiusX = filterWidth / 2;
            int filterRadiusY = filterHeight / 2;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float sum = 0.0f;

                    // Apply the filter
                    for (int fy = 0; fy < filterHeight; fy++)
                    {
                        int inputY = y + fy - filterRadiusY;
                        
                        // Handle boundary conditions (clamp to edge)
                        if (inputY < 0) inputY = 0;
                        if (inputY >= height) inputY = height - 1;

                        for (int fx = 0; fx < filterWidth; fx++)
                        {
                            int inputX = x + fx - filterRadiusX;
                            
                            // Handle boundary conditions (clamp to edge)
                            if (inputX < 0) inputX = 0;
                            if (inputX >= width) inputX = width - 1;

                            float filterValue = filter[fy * filterWidth + fx];
                            float inputValue = input[inputY * width + inputX];
                            sum += filterValue * inputValue;
                        }
                    }

                    output[y * width + x] = sum;
                }
            }
        }
    }
} 