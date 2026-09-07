using System;
using System.Drawing;
using Emgu.CV;

namespace VehicleDetection
{
    /// <summary>
    /// A source of vehicle bounding boxes for one frame.
    /// </summary>
    /// <remarks>
    /// <see cref="TrafficVehicleDetection"/> finds them by background subtraction, which needs no
    /// model file and no training but only sees what moves. <see cref="YoloVehicleDetection"/> runs
    /// a neural network, which recognises a vehicle whether it moves or not, at the price of a
    /// model file and more CPU. Both feed the same <see cref="Engine"/> and <see cref="VehicleTracker"/>.
    /// </remarks>
    public interface IVehicleDetector : IDisposable
    {
        /// <summary>
        /// Detection zone expressed as fractions of the frame (0.0-1.0). Empty means full frame.
        /// </summary>
        RectangleF DetectionScale { get; set; }

        /// <summary>Detection zone in pixels, derived from <see cref="DetectionScale"/>.</summary>
        Rectangle DetectionZone { get; }

        /// <summary>
        /// When true a vehicle only counts if its box lies entirely inside the detection zone;
        /// otherwise its centre point has to be inside.
        /// </summary>
        bool RequireFullyInsideZone { get; set; }

        /// <summary>
        /// Intermediate image the detector works on, shown in the second window. Detectors that
        /// have nothing to show return an empty <see cref="Mat"/>.
        /// </summary>
        Mat OutputFrame { get; }

        /// <summary>Short name of the detector, for the console and the on screen display.</summary>
        string Name { get; }

        /// <summary>Returns the bounding boxes of the vehicles in the frame.</summary>
        Rectangle[] Detect(Mat inputFrame);

        /// <summary>Drops whatever state the detector accumulated, e.g. after the video restarts.</summary>
        void Reset();
    }
}
