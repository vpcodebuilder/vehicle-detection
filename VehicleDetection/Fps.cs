using System;
using System.Diagnostics;

namespace VehicleDetection
{
    /// <summary>
    /// Measures the real throughput of the processing loop.
    /// </summary>
    /// <remarks>
    /// The previous implementation used a <see cref="System.Timers.Timer"/> that mutated the
    /// counters from a thread pool thread while the capture thread was writing to them. This one
    /// is driven by the loop itself, so no synchronisation is required and no timer has to be
    /// disposed.
    /// </remarks>
    public sealed class Fps
    {
        /// <summary>Weight of the newest sample in the exponential moving average.</summary>
        private const double Smoothing = 0.1;

        private readonly Stopwatch stopwatch = new Stopwatch();
        private double lastTickMilliseconds;

        /// <summary>Number of frames measured since the last <see cref="Reset"/>.</summary>
        public long FrameCount { get; private set; }

        /// <summary>Frame rate of the most recent frame, in frames per second.</summary>
        public double CurrentFrameRate { get; private set; }

        /// <summary>Smoothed frame rate, in frames per second.</summary>
        public double AverageFrameRate { get; private set; }

        /// <summary>Wall clock time elapsed since the last <see cref="Reset"/>.</summary>
        public TimeSpan Elapsed
        {
            get { return stopwatch.Elapsed; }
        }

        /// <summary>Overall frame rate over the whole run, in frames per second.</summary>
        public double OverallFrameRate
        {
            get
            {
                double seconds = stopwatch.Elapsed.TotalSeconds;
                return seconds > 0.0 ? FrameCount / seconds : 0.0;
            }
        }

        /// <summary>Restarts the measurement.</summary>
        public void Reset()
        {
            FrameCount = 0;
            CurrentFrameRate = 0.0;
            AverageFrameRate = 0.0;
            lastTickMilliseconds = 0.0;
            stopwatch.Restart();
        }

        /// <summary>Registers a processed frame. Call it exactly once per loop iteration.</summary>
        public void Tick()
        {
            if (!stopwatch.IsRunning)
            {
                Reset();
                return;
            }

            double now = stopwatch.Elapsed.TotalMilliseconds;
            double delta = now - lastTickMilliseconds;
            lastTickMilliseconds = now;
            FrameCount++;

            if (delta <= 0.0) return;

            CurrentFrameRate = 1000.0 / delta;
            AverageFrameRate = AverageFrameRate <= 0.0
                ? CurrentFrameRate
                : AverageFrameRate + ((CurrentFrameRate - AverageFrameRate) * Smoothing);
        }
    }
}
