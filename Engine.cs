using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace VehicleDetection
{
    public class Engine
    {
        private const string RESULT_WINDOW_NAME = "result";
        private const string OUTPUT_WINDOW_NAME = "output";
        private const int ESC_KEY = 27;

        private readonly bool isCameraMode;
        private VideoCapture capture;        
        private Fps fps;
        private VideoInformation videoInfo;
        private TrafficVehicleDetection trafficVehicleDetection;

        private MCvScalar blueColor = new MCvScalar(255, 0, 0);
        private MCvScalar redColor = new MCvScalar(0, 0, 255);
        private MCvScalar whiteColor = new MCvScalar(255, 255, 255);
        private MCvScalar fontColor = new MCvScalar(0, 0, 0);
        private Mat inputFrame;
        private int frameCounter;
        
        /// <summary>
        /// Capture the frame in camera mode.
        /// </summary>
        /// <param name="captureIndex">camera index of USB port.</param>
        public Engine(int captureIndex = 0)
        {
            try
            {
                capture = new VideoCapture(captureIndex);
                capture.ImageGrabbed += ProcessFrame;
            }
            catch (Exception ex)
            {
                throw ex;
            }

            fps = new Fps(capture.GetCaptureProperty(CapProp.Fps));
            isCameraMode = true;
        }

        /// <summary>
        /// Using video file for play the media.
        /// </summary>
        /// <param name="videoPathFileName"></param>
        public Engine(string videoPathFileName)
        {
            try
            {
                capture = new VideoCapture(videoPathFileName);
                videoInfo = new VideoInformation(capture);
                capture.ImageGrabbed += ProcessFrame;
            }
            catch (Exception ex)
            {
                throw ex;
            }

            isCameraMode = false;
        }

        public void Run()
        {
            CvInvoke.NamedWindow(RESULT_WINDOW_NAME);
            CvInvoke.NamedWindow(OUTPUT_WINDOW_NAME);
            
            trafficVehicleDetection = new TrafficVehicleDetection();

            // Create the inputFrame.
            inputFrame = new Mat();
            capture.Start();

            if (isCameraMode)
            {
                fps.Start();
            }
            else
            {
                frameCounter = 0;
            }

            while (CvInvoke.WaitKey() == ESC_KEY)
            {
                capture.ImageGrabbed -= ProcessFrame;
                capture.Stop();
                CvInvoke.DestroyAllWindows();
                inputFrame.Dispose();
                break;
            }
        }
                        
        private void ProcessFrame(object sender, EventArgs e)
        {
            capture.Retrieve(inputFrame);

            if (isCameraMode)
            {
                fps.UpdateFrameCount();
            }
            else
            {
                frameCounter++;
            }

            if (inputFrame == null) return;
            
            // Resize the ratio of image. To speed processing mode.
            int resizeWidth = VideoInformation.TypeOfScreen(inputFrame) == ScreenType.FullScreen ? 640 : 960;

            while (inputFrame.Width > resizeWidth)
            {
                CvInvoke.PyrDown(inputFrame, inputFrame);
            }

            // Clone the inputFrame to resultFrame.
            Mat resultFrame = inputFrame.Clone();

            // Create the detection zone in rectangle of frame.
            Rectangle detectionZone = new Rectangle(0, inputFrame.Height / 2, inputFrame.Width, inputFrame.Height / 2);

            // Algorithms for vehicle detection.
            Rectangle[] rects = trafficVehicleDetection.Detect(inputFrame, detectionZone);
            
            // Draw title bar.
            CvInvoke.Rectangle(trafficVehicleDetection.OutputFrame, new Rectangle(0, 0, trafficVehicleDetection.OutputFrame.Width, 20), whiteColor, -1);

            // Draw text.
            if (isCameraMode)
            {
                CvInvoke.PutText(trafficVehicleDetection.OutputFrame,
                    string.Format("Fps: {0}", fps.AverageFrameRate.ToString("N2")),
                    new Point(5, 15),
                    FontFace.HersheyPlain,
                    1.0,
                    fontColor);
            }
            else
            {
                CvInvoke.PutText(trafficVehicleDetection.OutputFrame,
                    string.Format("Codec: {0} Frame: {1}/{2}", videoInfo.Codec, frameCounter, videoInfo.TotalFrames),
                    new Point(5, 15),
                    FontFace.HersheyPlain,
                    1.0,
                    fontColor);
            }
            
            // Draw detection zone.
            CvInvoke.Rectangle(resultFrame, detectionZone, blueColor, 2);

            // Draw center point of each rectangle.
            foreach (Rectangle rect in rects)
            {
                Point centerPoint = new Point(rect.X + (rect.Width / 2), rect.Y + rect.Height / 2);
                CvInvoke.Circle(resultFrame, centerPoint, 3, redColor, -1);
            }

            // Show image.
            CvInvoke.Imshow(RESULT_WINDOW_NAME, resultFrame);
            CvInvoke.Imshow(OUTPUT_WINDOW_NAME, trafficVehicleDetection.OutputFrame);
        }
    }
}
