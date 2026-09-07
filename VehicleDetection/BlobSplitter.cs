using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace VehicleDetection
{
    /// <summary>One region of a foreground blob, with the pixel count that produced it.</summary>
    public struct BlobPart
    {
        public readonly Rectangle Box;
        public readonly int Area;

        /// <summary>
        /// Standard deviation of the grey values inside the region. A vehicle has windows, pillars
        /// and wheels and scores high; its shadow on the asphalt is nearly flat and scores low.
        /// </summary>
        public readonly double Texture;

        public BlobPart(Rectangle box, int area, double texture)
        {
            Box = box;
            Area = area;
            Texture = texture;
        }
    }

    /// <summary>
    /// Splits a foreground blob that holds more than one vehicle back into its parts.
    /// </summary>
    /// <remarks>
    /// Vehicles driving side by side in neighbouring lanes touch in the mask, so background
    /// subtraction reports a single wide contour for all of them. The distance transform of such a
    /// blob has one local maximum per vehicle with a narrow neck in between, which is exactly what
    /// a marker controlled watershed needs: the maxima become the seeds, the watershed then cuts
    /// the blob along the necks.
    /// </remarks>
    public sealed class BlobSplitter : IDisposable
    {
        private static readonly Point DefaultAnchor = new Point(-1, -1);

        private readonly Mat blobMask = new Mat();
        private readonly Mat distance = new Mat();
        private readonly Mat seedsFloat = new Mat();
        private readonly Mat seeds = new Mat();
        private readonly Mat sureBackground = new Mat();
        private readonly Mat grayRoi = new Mat();
        private readonly List<BlobPart> candidates = new List<BlobPart>();

        private bool disposed;

        /// <summary>
        /// Fraction of the largest distance transform value a pixel needs to become a seed. Higher
        /// values keep only the cores of the vehicles, so they separate more easily but a lightly
        /// merged pair may be missed.
        /// </summary>
        public double SeedRatio { get; set; }

        /// <summary>Smallest seed that counts, as a fraction of the blob area.</summary>
        public double MinimumSeedAreaRatio { get; set; }

        /// <summary>Blobs whose distance transform peaks below this (in pixels) are never split.</summary>
        public double MinimumBlobThickness { get; set; }

        /// <summary>
        /// Share of the strongest region texture a region has to reach to survive. Splitting a
        /// vehicle off its own cast shadow is the common failure of this technique, and the shadow
        /// half is always the flat one.
        /// </summary>
        public double MinimumRelativeTexture { get; set; }

        /// <summary>Texture below which a region is dropped regardless of the other regions.</summary>
        public double MinimumTexture { get; set; }

        public BlobSplitter()
        {
            SeedRatio = 0.5;
            MinimumSeedAreaRatio = 0.06;
            MinimumBlobThickness = 4.0;
            MinimumRelativeTexture = 0.45;
            MinimumTexture = 15.0;
        }

        /// <summary>
        /// Tries to cut the contour <paramref name="contourIndex"/> into separate vehicles.
        /// </summary>
        /// <param name="colorFrame">The current frame in BGR, used as the watershed relief.</param>
        /// <param name="contours">All contours found in the mask.</param>
        /// <param name="contourIndex">Index of the blob to split.</param>
        /// <param name="box">Bounding box of that blob.</param>
        /// <param name="parts">Receives the regions the blob was cut into.</param>
        /// <returns>
        /// True when the blob was cut and <paramref name="parts"/> holds at least one region. A
        /// single region means the rest of the blob was cast shadow, so the caller still gets a
        /// tighter box than the whole blob would have been.
        /// </returns>
        public bool TrySplit(Mat colorFrame, VectorOfVectorOfPoint contours, int contourIndex,
            Rectangle box, List<BlobPart> parts)
        {
            if (disposed) throw new ObjectDisposedException("BlobSplitter");

            parts.Clear();

            // One pixel of slack all around, so the blob never touches the ROI border and the
            // distance transform sees the real shape.
            Rectangle roi = Rectangle.Intersect(
                new Rectangle(box.X - 2, box.Y - 2, box.Width + 4, box.Height + 4),
                new Rectangle(0, 0, colorFrame.Width, colorFrame.Height));

            if (roi.Width < 8 || roi.Height < 8) return false;

            RedrawBlobMask(contours, contourIndex, roi);

            int blobArea = CvInvoke.CountNonZero(blobMask);
            if (blobArea <= 0) return false;

            CvInvoke.DistanceTransform(blobMask, distance, null, DistType.L2, 3, DistLabelType.CComp);

            double minimum = 0.0;
            double maximum = 0.0;
            Point minimumLocation = Point.Empty;
            Point maximumLocation = Point.Empty;
            CvInvoke.MinMaxLoc(distance, ref minimum, ref maximum, ref minimumLocation, ref maximumLocation);

            if (maximum < MinimumBlobThickness) return false;

            CvInvoke.Threshold(distance, seedsFloat, maximum * SeedRatio, 255.0, ThresholdType.Binary);
            seedsFloat.ConvertTo(seeds, DepthType.Cv8U);

            using (Matrix<int> labels = new Matrix<int>(roi.Height, roi.Width))
            using (Mat stats = new Mat())
            using (Mat centroids = new Mat())
            {
                int labelCount = CvInvoke.ConnectedComponentsWithStats(seeds, labels, stats, centroids,
                    LineType.EightConnected, DepthType.Cv32S,
                    ConnectedComponentsAlgorithmsTypes.Default);

                // Label 0 is the background, so anything below three labels cannot be a merge.
                if (labelCount < 3) return false;

                int minimumSeedArea = Math.Max(16, (int)(blobArea * MinimumSeedAreaRatio));
                int[] markerOfLabel = new int[labelCount];
                int seedCount = 0;

                using (Matrix<int> statistics = new Matrix<int>(stats.Rows, stats.Cols))
                {
                    stats.CopyTo(statistics, null);

                    // Column 4 of the stats matrix is the component area.
                    for (int label = 1; label < labelCount; label++)
                    {
                        if (statistics.Data[label, 4] < minimumSeedArea) continue;

                        // Watershed reserves 1 for the background, so the seeds start at 2.
                        markerOfLabel[label] = ++seedCount + 1;
                    }
                }

                if (seedCount < 2) return false;

                using (Matrix<int> markers = BuildMarkers(labels, markerOfLabel, roi))
                using (Mat colorRoi = new Mat(colorFrame, roi))
                {
                    CvInvoke.CvtColor(colorRoi, grayRoi, ColorConversion.Bgr2Gray);
                    CvInvoke.Watershed(colorRoi, markers);

                    using (Matrix<byte> gray = new Matrix<byte>(roi.Height, roi.Width))
                    {
                        grayRoi.CopyTo(gray, null);
                        CollectParts(markers, gray, roi, seedCount, candidates);
                    }

                    DropShadowParts(candidates, parts);
                }
            }

            return parts.Count >= 1;
        }

        /// <summary>Paints only the requested contour into <see cref="blobMask"/>.</summary>
        private void RedrawBlobMask(VectorOfVectorOfPoint contours, int contourIndex, Rectangle roi)
        {
            if (blobMask.IsEmpty || blobMask.Width != roi.Width || blobMask.Height != roi.Height)
            {
                blobMask.Create(roi.Height, roi.Width, DepthType.Cv8U, 1);
            }

            blobMask.SetTo(new MCvScalar(0));

            CvInvoke.DrawContours(blobMask, contours, contourIndex, new MCvScalar(255), -1,
                LineType.EightConnected, null, int.MaxValue, new Point(-roi.X, -roi.Y));
        }

        /// <summary>
        /// Builds the watershed marker image: 1 for pixels that are certainly background, 2..n for
        /// the seed of each vehicle, 0 for the pixels the watershed has to assign.
        /// </summary>
        private Matrix<int> BuildMarkers(Matrix<int> labels, int[] markerOfLabel, Rectangle roi)
        {
            // Everything outside the slightly grown blob is certainly background.
            CvInvoke.Dilate(blobMask, sureBackground, null, DefaultAnchor, 2, BorderType.Constant,
                CvInvoke.MorphologyDefaultBorderValue);

            Matrix<int> markers = new Matrix<int>(roi.Height, roi.Width);
            markers.SetZero();

            using (Matrix<byte> background = new Matrix<byte>(roi.Height, roi.Width))
            {
                sureBackground.CopyTo(background, null);

                int[,] markerData = markers.Data;
                int[,] labelData = labels.Data;
                byte[,] backgroundData = background.Data;

                for (int y = 0; y < roi.Height; y++)
                {
                    for (int x = 0; x < roi.Width; x++)
                    {
                        int label = labelData[y, x];

                        if (label > 0 && markerOfLabel[label] > 0)
                        {
                            markerData[y, x] = markerOfLabel[label];
                        }
                        else if (backgroundData[y, x] == 0)
                        {
                            markerData[y, x] = 1;
                        }
                    }
                }
            }

            return markers;
        }

        /// <summary>
        /// Turns the watershed result back into bounding boxes in frame coordinates and measures
        /// how much texture each region carries.
        /// </summary>
        private static void CollectParts(Matrix<int> markers, Matrix<byte> gray, Rectangle roi,
            int seedCount, List<BlobPart> parts)
        {
            parts.Clear();

            int[] left = new int[seedCount + 2];
            int[] top = new int[seedCount + 2];
            int[] right = new int[seedCount + 2];
            int[] bottom = new int[seedCount + 2];
            int[] area = new int[seedCount + 2];
            double[] sum = new double[seedCount + 2];
            double[] sumOfSquares = new double[seedCount + 2];

            for (int marker = 2; marker <= seedCount + 1; marker++)
            {
                left[marker] = int.MaxValue;
                top[marker] = int.MaxValue;
                right[marker] = int.MinValue;
                bottom[marker] = int.MinValue;
            }

            int[,] data = markers.Data;
            byte[,] grayData = gray.Data;

            for (int y = 0; y < roi.Height; y++)
            {
                for (int x = 0; x < roi.Width; x++)
                {
                    int marker = data[y, x];

                    // 1 is background and -1 marks the watershed lines between the regions.
                    if (marker < 2 || marker > seedCount + 1) continue;

                    if (x < left[marker]) left[marker] = x;
                    if (x > right[marker]) right[marker] = x;
                    if (y < top[marker]) top[marker] = y;
                    if (y > bottom[marker]) bottom[marker] = y;
                    area[marker]++;

                    double value = grayData[y, x];
                    sum[marker] += value;
                    sumOfSquares[marker] += value * value;
                }
            }

            for (int marker = 2; marker <= seedCount + 1; marker++)
            {
                if (area[marker] <= 0) continue;

                Rectangle box = new Rectangle(
                    roi.X + left[marker],
                    roi.Y + top[marker],
                    right[marker] - left[marker] + 1,
                    bottom[marker] - top[marker] + 1);

                double mean = sum[marker] / area[marker];
                double variance = (sumOfSquares[marker] / area[marker]) - (mean * mean);
                double texture = variance > 0.0 ? Math.Sqrt(variance) : 0.0;

                parts.Add(new BlobPart(box, area[marker], texture));
            }
        }

        /// <summary>
        /// Keeps the regions that look like vehicles and drops the flat ones, which are the cast
        /// shadow the watershed happily separated from the vehicle that throws it.
        /// </summary>
        private void DropShadowParts(List<BlobPart> found, List<BlobPart> kept)
        {
            double strongestTexture = 0.0;
            foreach (BlobPart part in found)
            {
                if (part.Texture > strongestTexture) strongestTexture = part.Texture;
            }

            double threshold = Math.Min(strongestTexture * MinimumRelativeTexture, MinimumTexture);

            foreach (BlobPart part in found)
            {
                if (part.Texture >= threshold) kept.Add(part);
            }
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;

            blobMask.Dispose();
            distance.Dispose();
            seedsFloat.Dispose();
            seeds.Dispose();
            sureBackground.Dispose();
            grayRoi.Dispose();
        }
    }
}
