using System;
using System.Drawing;
using System.Globalization;
using System.IO;

namespace VehicleDetection
{
    internal static class Program
    {
        /// <summary>Clip shipped with the project, copied next to the executable on build.</summary>
        private const string DefaultVideoFileName = "Relaxing_highway_traffic.mp4";

        private const string AssetsFolderName = "assets";

        private const string ModelsFolderName = "models";

        private static int Main(string[] args)
        {
            Options options;

            if (!Options.TryParse(args, out options))
            {
                PrintUsage();
                return 2;
            }

            if (options.ShowHelp)
            {
                PrintUsage();
                return 0;
            }

            try
            {
                using (IVehicleDetector detection = CreateDetector(options))
                using (Engine engine = CreateEngine(options))
                {
                    engine.LoopVideo = options.Loop;
                    engine.ProcessingWidth = options.ProcessingWidth;
                    engine.DetectEveryNthFrame = options.DetectEveryNthFrame;
                    engine.CountingLinePosition = options.CountingLinePosition;
                    engine.ShowMaskWindow = !options.HideMask;

                    engine.Run(detection);
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("Error: " + ex.Message);

                // The native side of Emgu.CV fails with a TypeInitializationException when the
                // OpenCV binaries cannot be loaded, and the inner exception is the useful one.
                if (ex.InnerException != null)
                {
                    Console.Error.WriteLine("Cause: " + ex.InnerException.Message);
                }

                return 1;
            }
        }

        /// <summary>
        /// Builds the detector the options ask for. In auto mode a YOLO model next to the
        /// executable wins, because it is the better detector; without one the program still runs
        /// on background subtraction alone.
        /// </summary>
        private static IVehicleDetector CreateDetector(Options options)
        {
            string modelPath = options.Detector == DetectorKind.BackgroundSubtraction
                ? null
                : ResolveModelPath(options.ModelPath);

            if (options.Detector == DetectorKind.Yolo && modelPath == null)
            {
                throw new FileNotFoundException(
                    "No .onnx model found. Put one in the models folder next to the executable, " +
                    "or pass --model <path>. See the README for where to get one.");
            }

            if (modelPath != null)
            {
                YoloVehicleDetection yolo = new YoloVehicleDetection(modelPath)
                {
                    DetectionScale = options.DetectionScale,
                    RequireFullyInsideZone = options.RequireFullyInsideZone
                };

                if (options.Confidence > 0.0) yolo.ConfidenceThreshold = (float)options.Confidence;
                return yolo;
            }

            return new TrafficVehicleDetection
            {
                DetectionScale = options.DetectionScale,
                RequireFullyInsideZone = options.RequireFullyInsideZone,
                SplitMergedBlobs = !options.NoSplit
            };
        }

        /// <summary>Finds the ONNX model, or returns null when there is none.</summary>
        private static string ResolveModelPath(string requested)
        {
            if (!string.IsNullOrEmpty(requested))
            {
                string absolute = Path.GetFullPath(requested);

                if (!File.Exists(absolute))
                {
                    throw new FileNotFoundException("Model file not found: " + absolute, absolute);
                }

                return absolute;
            }

            string folder = Path.Combine(AppContext.BaseDirectory, ModelsFolderName);
            if (!Directory.Exists(folder)) return null;

            string[] models = Directory.GetFiles(folder, "*.onnx");
            Array.Sort(models, StringComparer.OrdinalIgnoreCase);

            return models.Length > 0 ? models[0] : null;
        }

        private static Engine CreateEngine(Options options)
        {
            if (options.CameraIndex.HasValue) return new Engine(options.CameraIndex.Value);

            string videoPath = ResolveVideoPath(options.VideoPath);
            return new Engine(videoPath);
        }

        /// <summary>
        /// Finds the clip to play. The old code derived the path from
        /// <c>Environment.CurrentDirectory</c>, which only lined up when the program was started
        /// from <c>bin\Debug</c> by Visual Studio and broke everywhere else.
        /// </summary>
        private static string ResolveVideoPath(string requested)
        {
            if (!string.IsNullOrEmpty(requested))
            {
                string absolute = Path.GetFullPath(requested);
                if (File.Exists(absolute)) return absolute;

                throw new FileNotFoundException("Video file not found: " + absolute, absolute);
            }

            foreach (string candidate in DefaultVideoCandidates())
            {
                if (File.Exists(candidate)) return Path.GetFullPath(candidate);
            }

            throw new FileNotFoundException(
                "Could not find " + DefaultVideoFileName + " next to the executable. " +
                "Pass a clip explicitly with --video <path>.");
        }

        private static System.Collections.Generic.IEnumerable<string> DefaultVideoCandidates()
        {
            string baseDirectory = AppContext.BaseDirectory;

            yield return Path.Combine(baseDirectory, AssetsFolderName, DefaultVideoFileName);
            yield return Path.Combine(baseDirectory, DefaultVideoFileName);

            // Walk up from the output folder so "dotnet run" and a plain checkout both work.
            DirectoryInfo directory = new DirectoryInfo(baseDirectory);
            for (int i = 0; i < 6 && directory != null; i++)
            {
                yield return Path.Combine(directory.FullName, AssetsFolderName, DefaultVideoFileName);
                directory = directory.Parent;
            }
        }

        private static void PrintUsage()
        {
            Console.WriteLine("Vehicle Detection 2.0 - Emgu.CV traffic vehicle detector.");
            Console.WriteLine();
            Console.WriteLine("Usage: VehicleDetection [options]");
            Console.WriteLine();
            Console.WriteLine("  --detector <kind>  auto (default), yolo or bgs. auto picks yolo when a model is");
            Console.WriteLine("                     present, otherwise background subtraction.");
            Console.WriteLine("  --model <path>     ONNX model to use. Default: the first .onnx in models/.");
            Console.WriteLine("  --confidence <v>   YOLO score threshold, 0.0-1.0. Default 0.35.");
            Console.WriteLine("  --detect-every <n> Run the detector on every n-th frame only and let the tracker");
            Console.WriteLine("                     bridge the gap. Default 1. Use 2 to make YOLO real time on a CPU.");
            Console.WriteLine("  --video <path>     Play a video file (default: bundled " + DefaultVideoFileName + ").");
            Console.WriteLine("  --camera <index>   Capture from a USB camera instead, 0 based.");
            Console.WriteLine("  --zone x,y,w,h     Detection zone as fractions of the frame, 0.0-1.0.");
            Console.WriteLine("                     Default 0,0.5,1,0.5 (bottom half). Use 'full' for the whole frame.");
            Console.WriteLine("  --line <0.0-1.0>   Counting line inside the zone, or 'off'. Default 0.5.");
            Console.WriteLine("  --width <pixels>   Processing width. Default: 640 for 4:3, 960 for wide screen.");
            Console.WriteLine("  --strict-zone      Only count a vehicle fully inside the zone.");
            Console.WriteLine("  --no-split         Do not split blobs that hold several vehicles.");
            Console.WriteLine("  --no-loop          Stop at the end of the clip instead of replaying it.");
            Console.WriteLine("  --no-mask          Hide the foreground mask window.");
            Console.WriteLine("  -h, --help         Show this help.");
            Console.WriteLine();
            Console.WriteLine("Keys: [Esc/Q] quit  [Space] pause  [R] restart  [S] save a snapshot");
        }

        /// <summary>Which detector the program should run.</summary>
        private enum DetectorKind
        {
            Auto,
            Yolo,
            BackgroundSubtraction
        }

        /// <summary>Command line arguments of the program.</summary>
        private sealed class Options
        {
            public DetectorKind Detector = DetectorKind.Auto;
            public string ModelPath;
            public double Confidence;
            public int DetectEveryNthFrame = 1;
            public string VideoPath;
            public int? CameraIndex;
            public RectangleF DetectionScale = new RectangleF(0.0f, 0.5f, 1.0f, 0.5f);
            public double CountingLinePosition = 0.5;
            public int ProcessingWidth;
            public bool RequireFullyInsideZone;
            public bool NoSplit;
            public bool Loop = true;
            public bool HideMask;
            public bool ShowHelp;

            public static bool TryParse(string[] args, out Options options)
            {
                options = new Options();
                if (args == null) return true;

                for (int i = 0; i < args.Length; i++)
                {
                    string argument = args[i];

                    switch (argument)
                    {
                        case "-h":
                        case "--help":
                        case "/?":
                            options.ShowHelp = true;
                            return true;

                        case "--video":
                            if (!TryTakeValue(args, ref i, "--video", out options.VideoPath)) return false;
                            break;

                        case "--model":
                            if (!TryTakeValue(args, ref i, "--model", out options.ModelPath)) return false;
                            break;

                        case "--detector":
                        {
                            string value;
                            if (!TryTakeValue(args, ref i, "--detector", out value)) return false;

                            switch (value.ToLowerInvariant())
                            {
                                case "auto":
                                    options.Detector = DetectorKind.Auto;
                                    break;
                                case "yolo":
                                    options.Detector = DetectorKind.Yolo;
                                    break;
                                case "bgs":
                                    options.Detector = DetectorKind.BackgroundSubtraction;
                                    break;
                                default:
                                    Console.Error.WriteLine("--detector expects auto, yolo or bgs.");
                                    return false;
                            }

                            break;
                        }

                        case "--detect-every":
                        {
                            string value;
                            int every;
                            if (!TryTakeValue(args, ref i, "--detect-every", out value)) return false;
                            if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out every) ||
                                every < 1)
                            {
                                Console.Error.WriteLine("--detect-every expects a value of at least 1.");
                                return false;
                            }

                            options.DetectEveryNthFrame = every;
                            break;
                        }

                        case "--confidence":
                        {
                            string value;
                            double confidence;
                            if (!TryTakeValue(args, ref i, "--confidence", out value)) return false;
                            if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out confidence) ||
                                confidence <= 0.0 || confidence >= 1.0)
                            {
                                Console.Error.WriteLine("--confidence expects a value between 0.0 and 1.0.");
                                return false;
                            }

                            options.Confidence = confidence;
                            break;
                        }

                        case "--camera":
                        {
                            string value;
                            int index;
                            if (!TryTakeValue(args, ref i, "--camera", out value)) return false;
                            if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out index) ||
                                index < 0)
                            {
                                Console.Error.WriteLine("--camera expects a non negative index.");
                                return false;
                            }

                            options.CameraIndex = index;
                            break;
                        }

                        case "--zone":
                        {
                            string value;
                            if (!TryTakeValue(args, ref i, "--zone", out value)) return false;
                            if (!TryParseZone(value, out options.DetectionScale)) return false;
                            break;
                        }

                        case "--line":
                        {
                            string value;
                            if (!TryTakeValue(args, ref i, "--line", out value)) return false;

                            if (string.Equals(value, "off", StringComparison.OrdinalIgnoreCase))
                            {
                                options.CountingLinePosition = -1.0;
                                break;
                            }

                            double position;
                            if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out position) ||
                                position < 0.0 || position > 1.0)
                            {
                                Console.Error.WriteLine("--line expects a value between 0.0 and 1.0, or 'off'.");
                                return false;
                            }

                            options.CountingLinePosition = position;
                            break;
                        }

                        case "--width":
                        {
                            string value;
                            int width;
                            if (!TryTakeValue(args, ref i, "--width", out value)) return false;
                            if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out width) ||
                                width < 160)
                            {
                                Console.Error.WriteLine("--width expects a value of at least 160.");
                                return false;
                            }

                            options.ProcessingWidth = width;
                            break;
                        }

                        case "--strict-zone":
                            options.RequireFullyInsideZone = true;
                            break;

                        case "--no-split":
                            options.NoSplit = true;
                            break;

                        case "--no-loop":
                            options.Loop = false;
                            break;

                        case "--no-mask":
                            options.HideMask = true;
                            break;

                        default:
                            Console.Error.WriteLine("Unknown argument: " + argument);
                            return false;
                    }
                }

                if (options.CameraIndex.HasValue && !string.IsNullOrEmpty(options.VideoPath))
                {
                    Console.Error.WriteLine("--video and --camera cannot be combined.");
                    return false;
                }

                return true;
            }

            private static bool TryTakeValue(string[] args, ref int index, string name, out string value)
            {
                if (index + 1 >= args.Length)
                {
                    value = null;
                    Console.Error.WriteLine(name + " expects a value.");
                    return false;
                }

                value = args[++index];
                return true;
            }

            private static bool TryParseZone(string value, out RectangleF zone)
            {
                zone = RectangleF.Empty;

                if (string.Equals(value, "full", StringComparison.OrdinalIgnoreCase)) return true;

                string[] parts = value.Split(',');
                if (parts.Length != 4)
                {
                    Console.Error.WriteLine("--zone expects x,y,width,height as fractions, or 'full'.");
                    return false;
                }

                float[] numbers = new float[4];
                for (int i = 0; i < 4; i++)
                {
                    if (!float.TryParse(parts[i].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out numbers[i]) ||
                        numbers[i] < 0.0f || numbers[i] > 1.0f)
                    {
                        Console.Error.WriteLine("--zone values have to be between 0.0 and 1.0.");
                        return false;
                    }
                }

                if (numbers[2] <= 0.0f || numbers[3] <= 0.0f)
                {
                    Console.Error.WriteLine("--zone width and height have to be greater than 0.");
                    return false;
                }

                zone = new RectangleF(numbers[0], numbers[1], numbers[2], numbers[3]);
                return true;
            }
        }
    }
}
