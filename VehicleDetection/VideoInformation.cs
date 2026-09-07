using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace VehicleDetection
{
    public enum ScreenType
    {
        /// <summary>4:3 and squarer sources.</summary>
        FullScreen,

        /// <summary>16:9 and wider sources.</summary>
        WideScreen
    }

    /// <summary>
    /// Snapshot of the properties reported by a <see cref="VideoCapture"/> source.
    /// </summary>
    public sealed class VideoInformation
    {
        /// <summary>Anything with an aspect ratio below this is treated as full screen.</summary>
        private const double WideScreenAspectRatioThreshold = 1.5; // half way between 4:3 and 16:9.

        public int FrameWidth { get; private set; }
        public int FrameHeight { get; private set; }
        public double FrameRate { get; private set; }
        public int TotalFrames { get; private set; }
        public string Codec { get; private set; }
        public ScreenType ScreenType { get; private set; }

        /// <summary>Width divided by height, or 0 when the source size is unknown.</summary>
        public double AspectRatio
        {
            get { return FrameHeight > 0 ? FrameWidth / (double)FrameHeight : 0.0; }
        }

        /// <summary>Length of the clip, or <see cref="TimeSpan.Zero"/> for a live source.</summary>
        public TimeSpan Duration
        {
            get
            {
                if (TotalFrames <= 0 || FrameRate <= 0.0) return TimeSpan.Zero;
                return TimeSpan.FromSeconds(TotalFrames / FrameRate);
            }
        }

        public VideoInformation(VideoCapture capture)
        {
            if (capture == null) throw new ArgumentNullException("capture");

            // Emgu.CV 4.x replaced GetCaptureProperty/SetCaptureProperty with Get/Set.
            FrameWidth = ToInt32(capture.Get(CapProp.FrameWidth));
            FrameHeight = ToInt32(capture.Get(CapProp.FrameHeight));
            FrameRate = capture.Get(CapProp.Fps);
            TotalFrames = ToInt32(capture.Get(CapProp.FrameCount));
            Codec = DecodeFourCC(capture.Get(CapProp.FourCC));
            ScreenType = TypeOfScreen(FrameWidth, FrameHeight);

            if (double.IsNaN(FrameRate) || double.IsInfinity(FrameRate) || FrameRate <= 0.0)
            {
                FrameRate = 0.0;
            }
        }

        public static ScreenType TypeOfScreen(Mat frame)
        {
            if (frame == null || frame.IsEmpty) return ScreenType.WideScreen;
            return TypeOfScreen(frame.Width, frame.Height);
        }

        public static ScreenType TypeOfScreen(Size size)
        {
            return TypeOfScreen(size.Width, size.Height);
        }

        /// <summary>
        /// Classifies a frame size by aspect ratio. The original code compared
        /// <c>height / width</c> to <c>0.75</c> with <c>==</c>, so anything that was not exactly
        /// 4:3 (1280x718 for instance) silently fell through to wide screen.
        /// </summary>
        public static ScreenType TypeOfScreen(int width, int height)
        {
            if (width <= 0 || height <= 0) return ScreenType.WideScreen;
            return width / (double)height >= WideScreenAspectRatioThreshold
                ? ScreenType.WideScreen
                : ScreenType.FullScreen;
        }

        public override string ToString()
        {
            return string.Format(
                System.Globalization.CultureInfo.InvariantCulture,
                "{0}x{1} @ {2:N2} fps, {3} frames ({4}), codec {5}, {6}",
                FrameWidth, FrameHeight, FrameRate, TotalFrames,
                Duration.ToString(@"hh\:mm\:ss"), Codec, ScreenType);
        }

        private static int ToInt32(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0.0) return 0;
            return value >= int.MaxValue ? int.MaxValue : (int)value;
        }

        /// <summary>
        /// Turns the FourCC property into its four character code. Backends that do not report a
        /// codec return 0 or a negative value, which used to make <c>Convert.ToUInt32</c> throw
        /// an <see cref="OverflowException"/> before the first frame was ever shown.
        /// </summary>
        private static string DecodeFourCC(double fourCC)
        {
            if (double.IsNaN(fourCC) || double.IsInfinity(fourCC) || fourCC <= 0.0) return "n/a";

            uint code = unchecked((uint)(long)fourCC);
            char[] characters = new char[4];

            for (int i = 0; i < 4; i++)
            {
                char character = (char)((code >> (8 * i)) & 0xFF);
                characters[i] = char.IsLetterOrDigit(character) || char.IsPunctuation(character)
                    ? character
                    : ' ';
            }

            string codec = new string(characters).Trim();
            return codec.Length == 0 ? "n/a" : codec;
        }
    }
}
