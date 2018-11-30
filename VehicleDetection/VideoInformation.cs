using Emgu.CV;
using System;
using System.Text;

namespace VehicleDetection
{
    public enum ScreenType
    {
        FullScreen,
        WideScreen
    }

    public sealed class VideoInformation
    {
        public int FrameWidth;
        public int FrameHeight;
        public int FrameRate;
        public int TotalFrames;
        public string Codec;
        public ScreenType screenType;

        public VideoInformation(VideoCapture capture)
        {
            this.FrameWidth = (int)capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth);
            this.FrameHeight = (int)capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight);
            this.FrameRate = (int)capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps);
            this.TotalFrames = (int)capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameCount);

            double fourCC = capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FourCC);
            this.Codec = new string(Encoding.UTF8.GetString(BitConverter.GetBytes(Convert.ToUInt32(fourCC))).ToCharArray());

            if (FrameHeight / (double)FrameWidth == 0.75)
            {
                screenType = ScreenType.FullScreen;
            }
            else
            {
                screenType = ScreenType.WideScreen;
            }
        }

        public static ScreenType TypeOfScreen(Mat frame)
        {
            if (frame.Height / (double)frame.Width == 0.75)
            {
                return ScreenType.FullScreen;
            }
            else
            {
                return ScreenType.WideScreen;
            }
        }
    }
}
