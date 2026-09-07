using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace VehicleDetection
{
    /// <summary>
    /// Owns the capture source and drives the detect -> track -> render loop.
    /// </summary>
    /// <remarks>
    /// The frames are pulled synchronously on the thread that calls <see cref="Run"/>. The old
    /// version processed frames on the capture thread raised by <c>ImageGrabbed</c> while the main
    /// thread sat in a blocking <c>WaitKey()</c>; HighGUI is not thread safe, so the windows only
    /// repainted by accident and any key press ended the program instead of only Esc.
    /// </remarks>
    public sealed class Engine : IDisposable
    {
        private const string ResultWindowName = "result";
        private const string MaskWindowName = "output";
        private const int EscapeKey = 27;
        private const int HudHeight = 22;

        private static readonly MCvScalar BlueColor = new MCvScalar(255, 0, 0);
        private static readonly MCvScalar GreenColor = new MCvScalar(0, 200, 0);
        private static readonly MCvScalar RedColor = new MCvScalar(0, 0, 255);
        private static readonly MCvScalar YellowColor = new MCvScalar(0, 220, 220);
        private static readonly MCvScalar BlackColor = new MCvScalar(0, 0, 0);
        private static readonly MCvScalar WhiteColor = new MCvScalar(255, 255, 255);

        private readonly bool isCameraMode;
        private readonly string sourceName;
        private readonly VideoCapture capture;
        private readonly VideoInformation videoInfo;
        private readonly Fps fps = new Fps();
        private readonly VehicleTracker tracker = new VehicleTracker();

        private readonly Mat inputFrame = new Mat();
        private readonly Mat resultFrame = new Mat();
        private readonly Mat maskView = new Mat();

        private int frameCounter;
        private int countingLineY = -1;
        private bool disposed;

        /// <summary>Replays the clip from the start when it ends. Ignored for cameras.</summary>
        public bool LoopVideo { get; set; }

        /// <summary>
        /// Width the frames are scaled down to before processing, or 0 to pick it from the aspect
        /// ratio (640 px for 4:3 sources, 960 px for wide screen).
        /// </summary>
        public int ProcessingWidth { get; set; }

        /// <summary>
        /// Position of the counting line inside the detection zone, as a fraction of its height.
        /// Negative disables counting.
        /// </summary>
        public double CountingLinePosition { get; set; }

        /// <summary>Shows the binary foreground mask in a second window.</summary>
        public bool ShowMaskWindow { get; set; }

        /// <summary>
        /// Runs the detector on every n-th frame only and lets the tracker hold the boxes in
        /// between. A neural detector that needs 60 ms per frame cannot keep up with a 24 fps
        /// source; detecting every second frame halves its cost, and the tracker keeps the
        /// identities and the counters intact across the gap.
        /// </summary>
        public int DetectEveryNthFrame { get; set; }

        public VideoInformation VideoInformation
        {
            get { return videoInfo; }
        }

        public VehicleTracker Tracker
        {
            get { return tracker; }
        }

        /// <summary>Captures from a USB camera.</summary>
        /// <param name="captureIndex">Zero based camera index.</param>
        public Engine(int captureIndex)
        {
            isCameraMode = true;
            sourceName = "camera #" + captureIndex.ToString(CultureInfo.InvariantCulture);
            capture = new VideoCapture(captureIndex, VideoCapture.API.Any);

            if (!capture.IsOpened)
            {
                capture.Dispose();
                throw new InvalidOperationException("Unable to open " + sourceName + ".");
            }

            videoInfo = new VideoInformation(capture);
            ApplyDefaults();
        }

        /// <summary>Plays a video file.</summary>
        public Engine(string videoPathFileName)
        {
            if (string.IsNullOrEmpty(videoPathFileName))
            {
                throw new ArgumentException("A video path is required.", "videoPathFileName");
            }

            if (!File.Exists(videoPathFileName))
            {
                throw new FileNotFoundException("Video file not found: " + videoPathFileName, videoPathFileName);
            }

            isCameraMode = false;
            sourceName = Path.GetFileName(videoPathFileName);
            capture = new VideoCapture(videoPathFileName, VideoCapture.API.Any);

            if (!capture.IsOpened)
            {
                capture.Dispose();
                throw new InvalidOperationException(
                    "Unable to decode " + videoPathFileName + ". The codec may not be supported.");
            }

            videoInfo = new VideoInformation(capture);
            ApplyDefaults();
        }

        private void ApplyDefaults()
        {
            LoopVideo = !isCameraMode;
            ProcessingWidth = 0;
            DetectEveryNthFrame = 1;
            CountingLinePosition = 0.5;
            ShowMaskWindow = true;
        }

        /// <summary>
        /// Runs until the user quits, the camera fails or the clip ends. The detector is not
        /// disposed here; the caller owns it.
        /// </summary>
        public void Run(IVehicleDetector detection)
        {
            if (detection == null) throw new ArgumentNullException("detection");
            if (disposed) throw new ObjectDisposedException("Engine");

            Console.WriteLine("Source : {0}", sourceName);
            Console.WriteLine("Stream : {0}", videoInfo);
            Console.WriteLine("Engine : {0}", detection.Name);
            Console.WriteLine("Keys   : [Esc/Q] quit  [Space] pause  [R] restart  [S] snapshot");

            CvInvoke.NamedWindow(ResultWindowName, WindowFlags.AutoSize);
            bool maskWindowCreated = false;

            Stopwatch frameTimer = new Stopwatch();
            bool paused = false;
            bool running = true;
            int consecutiveReadFailures = 0;

            frameCounter = 0;
            tracker.Reset();
            fps.Reset();

            while (running)
            {
                frameTimer.Restart();

                if (!paused)
                {
                    if (!capture.Read(inputFrame) || inputFrame.IsEmpty)
                    {
                        if (isCameraMode)
                        {
                            // A camera can drop a frame now and then; give up only if it keeps failing.
                            if (++consecutiveReadFailures < 30) continue;
                            Console.WriteLine("The camera stopped delivering frames.");
                            break;
                        }

                        if (LoopVideo && frameCounter > 0)
                        {
                            Restart(detection);
                            continue;
                        }

                        Console.WriteLine("End of stream.");
                        break;
                    }

                    consecutiveReadFailures = 0;
                    frameCounter++;

                    // Scale down for speed. PyrDown halves the frame, so a final resize is needed
                    // to hit the exact target width.
                    FitToWidth(inputFrame, ResolveProcessingWidth(inputFrame));

                    if (IsDetectionFrame())
                    {
                        Rectangle[] boxes = detection.Detect(inputFrame);

                        UpdateCountingLine(detection, inputFrame.Size);
                        tracker.Update(boxes, inputFrame.Size, countingLineY);
                    }

                    fps.Tick();

                    RenderResult(detection);
                    if (ShowMaskWindow) RenderMask(detection);
                }

                if (!resultFrame.IsEmpty)
                {
                    CvInvoke.Imshow(ResultWindowName, resultFrame);
                }

                // Detectors without an intermediate image never open the second window.
                if (ShowMaskWindow && !maskView.IsEmpty)
                {
                    if (!maskWindowCreated)
                    {
                        CvInvoke.NamedWindow(MaskWindowName, WindowFlags.AutoSize);
                        maskWindowCreated = true;
                    }

                    CvInvoke.Imshow(MaskWindowName, maskView);
                }

                int key = CvInvoke.WaitKey(NextDelay(frameTimer));
                if (key >= 0) key &= 0xFF;

                switch (key)
                {
                    case EscapeKey:
                    case 'q':
                    case 'Q':
                        running = false;
                        break;

                    case ' ':
                        paused = !paused;
                        break;

                    case 'r':
                    case 'R':
                        Restart(detection);
                        paused = false;
                        break;

                    case 's':
                    case 'S':
                        SaveSnapshot();
                        break;
                }

                if (running && IsWindowClosed(ResultWindowName)) running = false;
            }

            CvInvoke.DestroyAllWindows();
            PrintSummary();
        }

        /// <summary>True when the detector has to run on the frame just read.</summary>
        private bool IsDetectionFrame()
        {
            if (DetectEveryNthFrame <= 1) return true;

            // frameCounter is 1 based, so the first frame of every group is a detection frame.
            return (frameCounter - 1) % DetectEveryNthFrame == 0;
        }

        private void PrintSummary()
        {
            Console.WriteLine();
            Console.WriteLine("Frames processed : {0}", frameCounter);
            Console.WriteLine("Processing rate  : {0:N2} fps", fps.OverallFrameRate);
            Console.WriteLine("Vehicles counted : {0} (down {1}, up {2})",
                tracker.TotalCount, tracker.CountedDownwards, tracker.CountedUpwards);
        }

        private void Restart(IVehicleDetector detection)
        {
            if (isCameraMode) return;

            capture.Set(CapProp.PosFrames, 0);
            frameCounter = 0;
            tracker.Reset();
            detection.Reset();
            fps.Reset();
        }

        /// <summary>Milliseconds to hand to WaitKey so a clip plays at its native speed.</summary>
        private int NextDelay(Stopwatch frameTimer)
        {
            if (isCameraMode || videoInfo.FrameRate <= 0.0) return 1;

            double budget = 1000.0 / videoInfo.FrameRate;
            double spent = frameTimer.Elapsed.TotalMilliseconds;
            int delay = (int)Math.Round(budget - spent);

            return delay < 1 ? 1 : delay;
        }

        private int ResolveProcessingWidth(Mat frame)
        {
            if (ProcessingWidth > 0) return ProcessingWidth;
            return VideoInformation.TypeOfScreen(frame) == ScreenType.FullScreen ? 640 : 960;
        }

        private void UpdateCountingLine(IVehicleDetector detection, Size frameSize)
        {
            if (CountingLinePosition < 0.0)
            {
                countingLineY = -1;
                return;
            }

            Rectangle zone = detection.DetectionZone == Rectangle.Empty
                ? new Rectangle(0, 0, frameSize.Width, frameSize.Height)
                : detection.DetectionZone;

            countingLineY = zone.Y + (int)Math.Round(zone.Height * CountingLinePosition);
        }

        private void RenderResult(IVehicleDetector detection)
        {
            inputFrame.CopyTo(resultFrame);

            if (detection.DetectionZone != Rectangle.Empty)
            {
                CvInvoke.Rectangle(resultFrame, detection.DetectionZone, BlueColor, 2);
            }

            if (countingLineY >= 0)
            {
                Rectangle zone = detection.DetectionZone == Rectangle.Empty
                    ? new Rectangle(0, 0, resultFrame.Width, resultFrame.Height)
                    : detection.DetectionZone;

                CvInvoke.Line(resultFrame,
                    new Point(zone.Left, countingLineY),
                    new Point(zone.Right, countingLineY),
                    YellowColor, 2, LineType.EightConnected, 0);
            }

            foreach (TrackedVehicle track in tracker.Tracks)
            {
                if (track.MissingFrames > 0) continue;

                MCvScalar boxColor = track.Counted ? RedColor : GreenColor;
                CvInvoke.Rectangle(resultFrame, track.Box, boxColor, 2);
                CvInvoke.Circle(resultFrame, track.Centroid, 3, RedColor, -1);

                CvInvoke.PutText(resultFrame,
                    "#" + track.Id.ToString(CultureInfo.InvariantCulture),
                    new Point(track.Box.X, Math.Max(10, track.Box.Y - 4)),
                    FontFace.HersheyPlain, 0.9, boxColor, 1, LineType.AntiAlias, false);

                DrawTrail(resultFrame, track);
            }

            DrawHud(resultFrame);
        }

        private static void DrawTrail(IInputOutputArray canvas, TrackedVehicle track)
        {
            IReadOnlyList<Point> trail = track.Trail;

            for (int i = 1; i < trail.Count; i++)
            {
                CvInvoke.Line(canvas, trail[i - 1], trail[i], GreenColor, 1, LineType.AntiAlias, 0);
            }
        }

        private void RenderMask(IVehicleDetector detection)
        {
            Mat mask = detection.OutputFrame;
            if (mask == null || mask.IsEmpty) return;

            // Colourise so the overlay is readable on top of the binary mask.
            CvInvoke.CvtColor(mask, maskView, ColorConversion.Gray2Bgr);

            if (detection.DetectionZone != Rectangle.Empty)
            {
                CvInvoke.Rectangle(maskView, detection.DetectionZone, BlueColor, 2);
            }

            DrawHud(maskView);
        }

        private void DrawHud(Mat canvas)
        {
            CvInvoke.Rectangle(canvas, new Rectangle(0, 0, canvas.Width, HudHeight), WhiteColor, -1);
            CvInvoke.PutText(canvas, BuildHudText(), new Point(5, 16),
                FontFace.HersheyPlain, 1.0, BlackColor, 1, LineType.AntiAlias, false);
        }

        private string BuildHudText()
        {
            string position = isCameraMode
                ? string.Format(CultureInfo.InvariantCulture, "Frame: {0}", frameCounter)
                : string.Format(CultureInfo.InvariantCulture, "Codec: {0}  Frame: {1}/{2}",
                    videoInfo.Codec, frameCounter, videoInfo.TotalFrames);

            return string.Format(CultureInfo.InvariantCulture,
                "{0}  Fps: {1:N1}  Vehicles: {2}",
                position, fps.AverageFrameRate, tracker.TotalCount);
        }

        private void SaveSnapshot()
        {
            if (resultFrame.IsEmpty) return;

            try
            {
                string folder = Path.Combine(AppContext.BaseDirectory, "snapshots");
                Directory.CreateDirectory(folder);

                string file = Path.Combine(folder, string.Format(CultureInfo.InvariantCulture,
                    "frame_{0:yyyyMMdd_HHmmss_fff}.png", DateTime.Now));

                CvInvoke.Imwrite(file, resultFrame);
                Console.WriteLine("Snapshot saved to {0}", file);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Could not save the snapshot: {0}", ex.Message);
            }
        }

        /// <summary>True when the user closed the window with its title bar button.</summary>
        private static bool IsWindowClosed(string windowName)
        {
            try
            {
                return CvInvoke.GetWindowProperty(windowName, WindowPropertyFlags.Visible) < 1.0;
            }
            catch (Exception)
            {
                return true;
            }
        }

        /// <summary>
        /// Scales <paramref name="frame"/> down in place so that it is at most
        /// <paramref name="targetWidth"/> wide, keeping the aspect ratio.
        /// </summary>
        private static void FitToWidth(Mat frame, int targetWidth)
        {
            if (targetWidth <= 0 || frame.Width <= targetWidth) return;

            while (frame.Width / 2 >= targetWidth && frame.Width > 2 && frame.Height > 2)
            {
                CvInvoke.PyrDown(frame, frame);
            }

            if (frame.Width <= targetWidth) return;

            int height = (int)Math.Round(frame.Height * (targetWidth / (double)frame.Width));
            CvInvoke.Resize(frame, frame, new Size(targetWidth, Math.Max(1, height)), 0, 0, Inter.Area);
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;

            capture.Dispose();
            inputFrame.Dispose();
            resultFrame.Dispose();
            maskView.Dispose();
        }
    }
}
