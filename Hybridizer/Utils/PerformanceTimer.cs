using System;
using System.Diagnostics;

namespace HybridizerSample.Utils
{
    /// <summary>
    /// Utility for measuring and comparing performance between CPU and GPU
    /// </summary>
    public class PerformanceTimer
    {
        private Stopwatch _stopwatch;
        private long _cpuElapsed;
        private long _gpuElapsed;
        private string _operationName;
        
        /// <summary>
        /// Initializes a new instance of the PerformanceTimer class
        /// </summary>
        /// <param name="operationName">Name of the operation being timed</param>
        public PerformanceTimer(string operationName)
        {
            _stopwatch = new Stopwatch();
            _operationName = operationName;
        }
        
        /// <summary>
        /// Gets or sets the CPU elapsed time in milliseconds
        /// </summary>
        public long CpuElapsedMs { get => _cpuElapsed; set => _cpuElapsed = value; }
        
        /// <summary>
        /// Gets the GPU elapsed time in milliseconds
        /// </summary>
        public long GpuElapsedMs => _gpuElapsed;
        
        /// <summary>
        /// Gets the speedup factor (CPU time / GPU time)
        /// </summary>
        public double Speedup => _gpuElapsed > 0 ? (double)_cpuElapsed / _gpuElapsed : 0;
        
        /// <summary>
        /// Starts the timer
        /// </summary>
        public void Start()
        {
            _stopwatch.Reset();
            _stopwatch.Start();
        }
        
        /// <summary>
        /// Stops the timer and records the elapsed time for the CPU implementation
        /// </summary>
        public void StopCpu()
        {
            _stopwatch.Stop();
            _cpuElapsed = _stopwatch.ElapsedMilliseconds;
        }
        
        /// <summary>
        /// Stops the timer and records the elapsed time for the GPU implementation
        /// </summary>
        public void StopGpu()
        {
            _stopwatch.Stop();
            _gpuElapsed = _stopwatch.ElapsedMilliseconds;
        }
        
        /// <summary>
        /// Prints performance comparison results
        /// </summary>
        public void PrintResults()
        {
            Console.WriteLine($"Performance Results for {_operationName}:");
            Console.WriteLine($"  CPU Time: {_cpuElapsed} ms");
            Console.WriteLine($"  GPU Time: {_gpuElapsed} ms");
            
            if (_gpuElapsed > 0)
            {
                Console.WriteLine($"  Speedup: {Speedup:F2}x");
            }
            else
            {
                Console.WriteLine("  Speedup: N/A (GPU not available)");
            }
            
            Console.WriteLine();
        }
    }
} 