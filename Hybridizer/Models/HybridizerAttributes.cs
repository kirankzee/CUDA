using System;

namespace HybridizerSample.Models
{
    /// <summary>
    /// Marks a method as an entry point for GPU execution
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class EntryPointAttribute : Attribute
    {
    }

    /// <summary>
    /// Configures the CUDA launch parameters for a kernel
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class HybridizeLaunchConfigAttribute : Attribute
    {
        /// <summary>
        /// Gets or sets the thread block dimensions
        /// </summary>
        public int[] Block { get; }

        /// <summary>
        /// Gets or sets the grid dimensions
        /// </summary>
        public int[]? Grid { get; set; }

        /// <summary>
        /// Gets or sets the shared memory size in bytes
        /// </summary>
        public int SharedMemorySize { get; set; }

        /// <summary>
        /// Initializes a new instance of the HybridizeLaunchConfigAttribute class
        /// </summary>
        /// <param name="block">Thread block dimensions</param>
        public HybridizeLaunchConfigAttribute(int[] block)
        {
            Block = block;
            Grid = null; // Auto-calculated by Hybridizer
            SharedMemorySize = 0;
        }
    }

    /// <summary>
    /// CUDA thread and block indices (would be provided by Hybridizer in real implementation)
    /// </summary>
    public static class threadIdx
    {
        public static int x { get; set; }
        public static int y { get; set; }
        public static int z { get; set; }
    }

    /// <summary>
    /// CUDA block indices (would be provided by Hybridizer in real implementation)
    /// </summary>
    public static class blockIdx
    {
        public static int x { get; set; }
        public static int y { get; set; }
        public static int z { get; set; }
    }

    /// <summary>
    /// CUDA block dimensions (would be provided by Hybridizer in real implementation)
    /// </summary>
    public static class blockDim
    {
        public static int x { get; set; }
        public static int y { get; set; }
        public static int z { get; set; }
    }

    /// <summary>
    /// CUDA grid dimensions (would be provided by Hybridizer in real implementation)
    /// </summary>
    public static class gridDim
    {
        public static int x { get; set; }
        public static int y { get; set; }
        public static int z { get; set; }
    }
} 