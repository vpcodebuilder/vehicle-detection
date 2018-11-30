using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.BgSegm;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace VehicleDetection
{
    public class TrafficVehicleDetection
    {
        public Mat OutputFrame;
        public RectangleF DetectionScale = RectangleF.Empty;
        public Rectangle DetectionZone = Rectangle.Empty;
        private readonly BackgroundSubtractorCNT cnt;

        public TrafficVehicleDetection()
        {
            OutputFrame = new Mat();
            cnt = new BackgroundSubtractorCNT();
        }

        public Rectangle[] Detect(Mat inputFrame)
        {
            // Convert BGR from originalFrame to grayscale outputFrame.
            CvInvoke.CvtColor(inputFrame, OutputFrame, ColorConversion.Bgr2Gray);

            // Blur noise reduction.
            CvInvoke.GaussianBlur(OutputFrame, OutputFrame, new Size(5, 5), 0);

            // Background subtraction CNT algorithms technic.
            cnt.Apply(OutputFrame, OutputFrame);

            // Blur after background subtraction for extract feature.
            CvInvoke.GaussianBlur(OutputFrame, OutputFrame, new Size(11, 11), 0);

            // Threshold binary from blur.
            CvInvoke.Threshold(OutputFrame, OutputFrame, 0, 255, ThresholdType.Binary);

            // Do erode for reduce size of feature around 10 times.
            CvInvoke.Erode(OutputFrame, OutputFrame, null, new Point(-1, -1), 10, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);

            List<Rectangle> rects = new List<Rectangle>();
            
            // Find contours in external shape.
            using (var contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(OutputFrame, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                
                var contourPointsOfPoints = contours.ToArrayOfArray();

                foreach (var contourPoints in contourPointsOfPoints)
                {
                    using (var approxContour = new VectorOfPoint())
                    {
                        VectorOfPoint contour = new VectorOfPoint(contourPoints);
                        var perimeter = CvInvoke.ArcLength(contour, true);
                        CvInvoke.ApproxPolyDP(contour, approxContour, 0.05 * perimeter, true);

                        if (approxContour == null) continue;

                        Rectangle rect = CvInvoke.BoundingRectangle(approxContour);

                        // now the contour have in range of detection zone?
                        if (DetectionScale == Rectangle.Empty)
                        {
                            rects.Add(rect);
                        }
                        else
                        {
                            if (DetectionZone == Rectangle.Empty)
                            {
                                RectangleF rectDetect = new RectangleF(
                                    inputFrame.Width * DetectionScale.X,
                                    inputFrame.Height * DetectionScale.Y,
                                    inputFrame.Width * DetectionScale.Width,
                                    inputFrame.Height * DetectionScale.Height);

                                DetectionZone = new Rectangle((int)rectDetect.X, (int)rectDetect.Y, (int)rectDetect.Width, (int)rectDetect.Height);
                            }

                            if (DetectionZone.Contains(rect))
                            {
                                rects.Add(rect);
                            }
                        }
                    }
                }
            }

            return rects.ToArray();
        }
    }
}
