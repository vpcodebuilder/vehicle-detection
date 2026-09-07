using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.BgSegm;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace VehicleDetection
{
    /// <summary>
    /// Extracts vehicle sized blobs from a traffic scene using background subtraction.
    /// </summary>
    /// <remarks>
    /// Pipeline: grayscale -> blur -> CNT background subtraction -> binary mask -> morphological
    /// open/close -> external contours -> splitting of merged blobs -> size and shape filtering ->
    /// detection zone filtering.
    /// </remarks>
    public sealed class TrafficVehicleDetection : IVehicleDetector
    {
        private static readonly Point DefaultAnchor = new Point(-1, -1);

        private readonly BackgroundSubtractorCNT subtractor;
        private readonly BlobSplitter splitter = new BlobSplitter();
        private readonly Mat grayFrame = new Mat();
        private readonly Mat colorFrame = new Mat();
        private readonly Mat foregroundMask = new Mat();
        private readonly List<Rectangle> detections = new List<Rectangle>();
        private readonly List<BlobPart> splitParts = new List<BlobPart>();

        private Mat openKernel;
        private Mat closeKernel;
        private Size kernelFrameSize = Size.Empty;
        private Size zoneFrameSize = Size.Empty;
        private bool disposed;

        /// <summary>
        /// Detection zone expressed as fractions of the frame (0.0-1.0). Empty means full frame.
        /// </summary>
        public RectangleF DetectionScale { get; set; }

        /// <summary>
        /// Detection zone in pixels, derived from <see cref="DetectionScale"/>. It is recomputed
        /// whenever the frame size changes instead of being cached forever after the first frame.
        /// </summary>
        public Rectangle DetectionZone { get; private set; }

        /// <summary>Binary foreground mask produced by the last <see cref="Detect"/> call.</summary>
        public Mat OutputFrame
        {
            get { return foregroundMask; }
        }

        public string Name
        {
            get { return "background subtraction (CNT)"; }
        }

        /// <summary>Smallest accepted blob, as a fraction of the frame area.</summary>
        public double MinimumAreaRatio { get; set; }

        /// <summary>Largest accepted blob, as a fraction of the frame area.</summary>
        public double MaximumAreaRatio { get; set; }

        /// <summary>Accepted width/height ratio range of a bounding box.</summary>
        public double MinimumAspectRatio { get; set; }

        public double MaximumAspectRatio { get; set; }

        /// <summary>
        /// Share of its bounding box a blob has to fill. A vehicle fills a good part of its box;
        /// a stretched shadow or a merge of unrelated blobs does not.
        /// </summary>
        public double MinimumFillRatio { get; set; }

        /// <summary>
        /// Cuts a blob that holds several vehicles back into one detection per vehicle. Vehicles
        /// driving side by side in neighbouring lanes touch in the mask and would otherwise be
        /// reported as a single wide box.
        /// </summary>
        public bool SplitMergedBlobs { get; set; }

        /// <summary>Tuning of the blob splitter, see <see cref="BlobSplitter"/>.</summary>
        public BlobSplitter Splitter
        {
            get { return splitter; }
        }

        /// <summary>Number of blobs the splitter reshaped during the last frame.</summary>
        public int SplitBlobCount { get; private set; }

        /// <summary>
        /// When true a blob only counts if its bounding box lies entirely inside the detection
        /// zone; otherwise its centre point has to be inside. The old code always required full
        /// containment, so every vehicle touching the zone border was dropped.
        /// </summary>
        public bool RequireFullyInsideZone { get; set; }

        /// <summary>Number of blobs rejected by the filters during the last frame.</summary>
        public int RejectedBlobCount { get; private set; }

        public TrafficVehicleDetection()
        {
            // minPixelStability: a pixel has to hold its value for ~half a second (15 frames at
            // 30 fps) before it becomes background; useHistory keeps long lived background stable.
            subtractor = new BackgroundSubtractorCNT(15, true, 15 * 60, true);

            DetectionScale = RectangleF.Empty;
            DetectionZone = Rectangle.Empty;
            MinimumAreaRatio = 0.0008;
            MaximumAreaRatio = 0.35;
            MinimumAspectRatio = 0.25;
            MaximumAspectRatio = 4.5;
            MinimumFillRatio = 0.35;
            SplitMergedBlobs = true;
            RequireFullyInsideZone = false;
        }

        /// <summary>
        /// Detects vehicle candidates in <paramref name="inputFrame"/> and returns their bounding
        /// boxes. The frame itself is never modified.
        /// </summary>
        public Rectangle[] Detect(Mat inputFrame)
        {
            if (disposed) throw new ObjectDisposedException("TrafficVehicleDetection");

            detections.Clear();
            RejectedBlobCount = 0;
            SplitBlobCount = 0;

            if (inputFrame == null || inputFrame.IsEmpty) return new Rectangle[0];

            Size frameSize = inputFrame.Size;
            EnsureKernels(frameSize);
            EnsureDetectionZone(frameSize);

            // Grayscale. Some sources are already single channel, and CvtColor would throw there.
            if (inputFrame.NumberOfChannels == 1)
            {
                inputFrame.CopyTo(grayFrame);
            }
            else
            {
                CvInvoke.CvtColor(inputFrame, grayFrame, ColorConversion.Bgr2Gray);
            }

            // Blur away sensor noise before the background model sees the frame.
            CvInvoke.GaussianBlur(grayFrame, grayFrame, new Size(5, 5), 0);

            // Background subtraction, CNT algorithm (fast and resilient to outdoor light changes).
            subtractor.Apply(grayFrame, foregroundMask);

            // Force a strictly binary mask. The old code blurred the mask and then thresholded at
            // 0, which promoted every faintly smeared pixel - including shadows - to foreground.
            CvInvoke.Threshold(foregroundMask, foregroundMask, 127, 255, ThresholdType.Binary);

            // Open removes speckle noise, close merges the parts of one vehicle into one blob.
            // The original used 10 erode iterations, which ate small cars entirely and shrank the
            // surviving boxes so much that their centre points no longer matched the vehicles.
            CvInvoke.MorphologyEx(foregroundMask, foregroundMask, MorphOp.Open, openKernel,
                DefaultAnchor, 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);
            CvInvoke.MorphologyEx(foregroundMask, foregroundMask, MorphOp.Close, closeKernel,
                DefaultAnchor, 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);

            double frameArea = (double)frameSize.Width * frameSize.Height;
            double minimumArea = frameArea * MinimumAreaRatio;
            double maximumArea = frameArea * MaximumAreaRatio;

            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(foregroundMask, contours, null, RetrType.External,
                    ChainApproxMethod.ChainApproxSimple);

                for (int i = 0; i < contours.Size; i++)
                {
                    double area;
                    Rectangle box;

                    using (VectorOfPoint contour = contours[i])
                    {
                        area = CvInvoke.ContourArea(contour, false);
                        box = CvInvoke.BoundingRectangle(contour);
                    }

                    // Too small to be a vehicle, or big enough to be a global light change.
                    if (area < minimumArea || area > maximumArea || box.Width <= 0 || box.Height <= 0)
                    {
                        RejectedBlobCount++;
                        continue;
                    }

                    int acceptedBefore = detections.Count;

                    // A blob with room for more than one vehicle goes to the splitter first.
                    if (SplitMergedBlobs && area >= minimumArea * 2.0 &&
                        splitter.TrySplit(EnsureColorFrame(inputFrame), contours, i, box, splitParts))
                    {
                        foreach (BlobPart part in splitParts)
                        {
                            TryAccept(part.Box, part.Area, minimumArea, maximumArea);
                        }

                        if (detections.Count > acceptedBefore)
                        {
                            SplitBlobCount++;
                            continue;
                        }
                    }

                    // Nothing usable came out of the split, so keep the blob whole.
                    if (!TryAccept(box, area, minimumArea, maximumArea)) RejectedBlobCount++;
                }
            }

            return detections.ToArray();
        }

        /// <summary>
        /// Applies the vehicle filters to one candidate box and keeps it when they all pass.
        /// </summary>
        /// <param name="area">Foreground pixel count of the candidate.</param>
        private bool TryAccept(Rectangle box, double area, double minimumArea, double maximumArea)
        {
            if (box.Width <= 0 || box.Height <= 0) return false;
            if (area < minimumArea || area > maximumArea) return false;

            double aspectRatio = box.Width / (double)box.Height;
            if (aspectRatio < MinimumAspectRatio || aspectRatio > MaximumAspectRatio) return false;

            if (area / (box.Width * (double)box.Height) < MinimumFillRatio) return false;

            if (!IsInsideDetectionZone(box)) return false;

            detections.Add(box);
            return true;
        }

        /// <summary>
        /// Returns the frame as BGR, which is what the watershed in the splitter needs as its
        /// relief. Colour sources are handed through untouched.
        /// </summary>
        private Mat EnsureColorFrame(Mat inputFrame)
        {
            if (inputFrame.NumberOfChannels == 3) return inputFrame;

            CvInvoke.CvtColor(inputFrame, colorFrame, ColorConversion.Gray2Bgr);
            return colorFrame;
        }

        /// <summary>Clears the mask, e.g. after the video restarts.</summary>
        public void Reset()
        {
            if (!foregroundMask.IsEmpty) foregroundMask.SetTo(new MCvScalar(0));
        }

        private bool IsInsideDetectionZone(Rectangle box)
        {
            if (DetectionZone == Rectangle.Empty) return true;

            if (RequireFullyInsideZone) return DetectionZone.Contains(box);

            Point center = new Point(box.X + (box.Width / 2), box.Y + (box.Height / 2));
            return DetectionZone.Contains(center);
        }

        private void EnsureDetectionZone(Size frameSize)
        {
            if (zoneFrameSize == frameSize) return;
            zoneFrameSize = frameSize;

            if (DetectionScale.IsEmpty || DetectionScale.Width <= 0.0f || DetectionScale.Height <= 0.0f)
            {
                DetectionZone = Rectangle.Empty;
                return;
            }

            int x = Clamp((int)Math.Round(frameSize.Width * DetectionScale.X), 0, frameSize.Width - 1);
            int y = Clamp((int)Math.Round(frameSize.Height * DetectionScale.Y), 0, frameSize.Height - 1);
            int width = Clamp((int)Math.Round(frameSize.Width * DetectionScale.Width), 1, frameSize.Width - x);
            int height = Clamp((int)Math.Round(frameSize.Height * DetectionScale.Height), 1, frameSize.Height - y);

            DetectionZone = new Rectangle(x, y, width, height);
        }

        /// <summary>
        /// Builds morphology kernels scaled to the frame, so the filtering behaves the same at
        /// 640 px and at 960 px.
        /// </summary>
        private void EnsureKernels(Size frameSize)
        {
            if (kernelFrameSize == frameSize && openKernel != null) return;
            kernelFrameSize = frameSize;

            // The close kernel is taller than it is wide on purpose. One vehicle breaks apart
            // vertically in the mask (roof, windscreen, body), while vehicles in neighbouring
            // lanes sit next to each other horizontally. A square kernel closed the gap between
            // the lanes as eagerly as the gap inside one vehicle and merged them into one blob.
            int closeHeight = Math.Max(5, (int)Math.Round(frameSize.Width / 90.0)) | 1; // keep it odd.
            int closeWidth = Math.Max(3, closeHeight / 2) | 1;

            if (openKernel != null) openKernel.Dispose();
            if (closeKernel != null) closeKernel.Dispose();

            openKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(3, 3), DefaultAnchor);
            closeKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(closeWidth, closeHeight), DefaultAnchor);
        }

        private static int Clamp(int value, int min, int max)
        {
            if (max < min) return min;
            return value < min ? min : (value > max ? max : value);
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;

            subtractor.Dispose();
            splitter.Dispose();
            grayFrame.Dispose();
            colorFrame.Dispose();
            foregroundMask.Dispose();
            if (openKernel != null) openKernel.Dispose();
            if (closeKernel != null) closeKernel.Dispose();
        }
    }
}
