using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VehicleDetection
{
    /// <summary>Shape of the tensor a YOLO export produces.</summary>
    public enum YoloOutputLayout
    {
        /// <summary>[1, boxes, 6] - x1, y1, x2, y2, score, class. YOLOv10, already deduplicated.</summary>
        EndToEnd,

        /// <summary>[1, 4 + classes, boxes] - cx, cy, w, h, class scores. YOLOv8 and YOLO11.</summary>
        Transposed,

        /// <summary>[1, boxes, 5 + classes] - cx, cy, w, h, objectness, class scores. YOLOv5 and v7.</summary>
        Legacy
    }

    /// <summary>
    /// Detects vehicles with a YOLO network through ONNX Runtime.
    /// </summary>
    /// <remarks>
    /// Background subtraction can only report foreground, so two vehicles that touch in the mask
    /// become one blob and a vehicle that stops disappears into the background. A detector
    /// recognises each vehicle on its own merit, which removes both problems: no merged blobs to
    /// split and no dependency on motion.
    /// </remarks>
    public sealed class YoloVehicleDetection : IVehicleDetector
    {
        /// <summary>COCO class ids of car, motorcycle, bus and truck.</summary>
        public static readonly int[] CocoVehicleClasses = { 2, 3, 5, 7 };

        private static readonly string[] CocoClassNames =
        {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        };

        private readonly InferenceSession session;
        private readonly string inputName;
        private readonly int inputWidth;
        private readonly int inputHeight;
        private readonly float[] inputBuffer;
        private readonly DenseTensor<float> inputTensor;
        private readonly List<NamedOnnxValue> inputs;

        private readonly Mat letterboxed = new Mat();
        private readonly Mat resized = new Mat();
        private readonly Mat emptyOutput = new Mat();
        private readonly List<Rectangle> detections = new List<Rectangle>();
        private readonly List<Candidate> candidates = new List<Candidate>();
        private readonly HashSet<int> acceptedClasses;

        private Size zoneFrameSize = Size.Empty;
        private bool disposed;

        public RectangleF DetectionScale { get; set; }

        public Rectangle DetectionZone { get; private set; }

        public bool RequireFullyInsideZone { get; set; }

        /// <summary>The network works on its own square input, so there is no mask to show.</summary>
        public Mat OutputFrame
        {
            get { return emptyOutput; }
        }

        public string Name { get; private set; }

        /// <summary>Smallest score a detection needs to be kept.</summary>
        public float ConfidenceThreshold { get; set; }

        /// <summary>Overlap above which two boxes of the same class are treated as one.</summary>
        public float NonMaximumSuppressionThreshold { get; set; }

        /// <summary>Layout the loaded model turned out to have.</summary>
        public YoloOutputLayout Layout { get; private set; }

        public YoloVehicleDetection(string modelPath)
            : this(modelPath, CocoVehicleClasses)
        {
        }

        /// <param name="modelPath">Path of the .onnx file.</param>
        /// <param name="classIds">Class ids to keep, or null to keep every class.</param>
        public YoloVehicleDetection(string modelPath, int[] classIds)
        {
            if (string.IsNullOrEmpty(modelPath)) throw new ArgumentException("A model path is required.", "modelPath");

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException("YOLO model not found: " + modelPath, modelPath);
            }

            DetectionScale = RectangleF.Empty;
            DetectionZone = Rectangle.Empty;
            ConfidenceThreshold = 0.35f;
            NonMaximumSuppressionThreshold = 0.45f;
            Name = "yolo:" + Path.GetFileNameWithoutExtension(modelPath);
            acceptedClasses = classIds == null || classIds.Length == 0 ? null : new HashSet<int>(classIds);

            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            session = new InferenceSession(modelPath, options);

            NodeMetadata input = session.InputMetadata.First().Value;
            inputName = session.InputMetadata.First().Key;

            // A dynamic axis is reported as a negative dimension; 640 is the YOLO default.
            inputHeight = input.Dimensions.Length == 4 && input.Dimensions[2] > 0 ? input.Dimensions[2] : 640;
            inputWidth = input.Dimensions.Length == 4 && input.Dimensions[3] > 0 ? input.Dimensions[3] : 640;

            inputBuffer = new float[3 * inputHeight * inputWidth];
            inputTensor = new DenseTensor<float>(inputBuffer, new[] { 1, 3, inputHeight, inputWidth });
            inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };
        }

        public Rectangle[] Detect(Mat inputFrame)
        {
            if (disposed) throw new ObjectDisposedException("YoloVehicleDetection");

            detections.Clear();
            candidates.Clear();

            if (inputFrame == null || inputFrame.IsEmpty) return new Rectangle[0];

            Size frameSize = inputFrame.Size;
            EnsureDetectionZone(frameSize);

            double scale = Letterbox(inputFrame);
            FillInputBuffer();

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs))
            {
                Tensor<float> output = results.First().AsTensor<float>();
                ReadOutput(output, scale, frameSize);
            }

            if (Layout != YoloOutputLayout.EndToEnd) SuppressOverlaps();

            foreach (Candidate candidate in candidates)
            {
                if (!candidate.Suppressed && IsInsideDetectionZone(candidate.Box))
                {
                    detections.Add(candidate.Box);
                }
            }

            return detections.ToArray();
        }

        /// <summary>Nothing is carried over between frames.</summary>
        public void Reset()
        {
        }

        /// <summary>Human readable name of a COCO class id.</summary>
        public static string ClassName(int classId)
        {
            return classId >= 0 && classId < CocoClassNames.Length
                ? CocoClassNames[classId]
                : classId.ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// Resizes the frame into the square network input while keeping its aspect ratio, padding
        /// the rest with the neutral grey the YOLO training pipeline uses. Stretching the frame
        /// instead would distort every vehicle and cost accuracy.
        /// </summary>
        /// <returns>The scale factor that was applied, to map the boxes back afterwards.</returns>
        private double Letterbox(Mat inputFrame)
        {
            double scale = Math.Min(inputWidth / (double)inputFrame.Width,
                                    inputHeight / (double)inputFrame.Height);

            int scaledWidth = Math.Max(1, (int)Math.Round(inputFrame.Width * scale));
            int scaledHeight = Math.Max(1, (int)Math.Round(inputFrame.Height * scale));

            CvInvoke.Resize(inputFrame, resized, new Size(scaledWidth, scaledHeight), 0, 0, Inter.Linear);

            if (letterboxed.IsEmpty || letterboxed.Width != inputWidth || letterboxed.Height != inputHeight)
            {
                letterboxed.Create(inputHeight, inputWidth, DepthType.Cv8U, 3);
            }

            letterboxed.SetTo(new MCvScalar(114, 114, 114));

            using (Mat target = new Mat(letterboxed, new Rectangle(0, 0, scaledWidth, scaledHeight)))
            {
                resized.CopyTo(target);
            }

            return scale;
        }

        /// <summary>Turns the letterboxed frame into the NCHW float tensor the network expects.</summary>
        private void FillInputBuffer()
        {
            // BlobFromImage does the scaling to 0..1, the BGR to RGB swap and the NCHW layout.
            using (Mat blob = DnnInvoke.BlobFromImage(letterboxed, 1.0 / 255.0,
                new Size(inputWidth, inputHeight), new MCvScalar(0, 0, 0), true, false, DepthType.Cv32F))
            {
                Marshal.Copy(blob.DataPointer, inputBuffer, 0, inputBuffer.Length);
            }
        }

        private void ReadOutput(Tensor<float> output, double scale, Size frameSize)
        {
            int[] dimensions = output.Dimensions.ToArray();

            if (dimensions.Length != 3)
            {
                throw new NotSupportedException(
                    "Unsupported YOLO output rank " + dimensions.Length + "; expected a rank 3 tensor.");
            }

            int rows = dimensions[1];
            int columns = dimensions[2];

            if (columns == 6 && rows >= columns)
            {
                Layout = YoloOutputLayout.EndToEnd;
                ReadEndToEnd(output, rows, scale, frameSize);
            }
            else if (rows < columns)
            {
                Layout = YoloOutputLayout.Transposed;
                ReadTransposed(output, rows, columns, scale, frameSize);
            }
            else
            {
                Layout = YoloOutputLayout.Legacy;
                ReadLegacy(output, rows, columns, scale, frameSize);
            }
        }

        /// <summary>YOLOv10: corners, score and class, already free of duplicates.</summary>
        private void ReadEndToEnd(Tensor<float> output, int rows, double scale, Size frameSize)
        {
            for (int row = 0; row < rows; row++)
            {
                float score = output[0, row, 4];

                // The rows come out sorted by score, so the first weak one ends the list.
                if (score < ConfidenceThreshold) break;

                int classId = (int)output[0, row, 5];
                if (!IsAcceptedClass(classId)) continue;

                float x1 = output[0, row, 0];
                float y1 = output[0, row, 1];
                float x2 = output[0, row, 2];
                float y2 = output[0, row, 3];

                AddCandidate(x1, y1, x2 - x1, y2 - y1, score, classId, scale, frameSize);
            }
        }

        /// <summary>YOLOv8 and YOLO11: one column per box, four box values then the class scores.</summary>
        private void ReadTransposed(Tensor<float> output, int rows, int columns, double scale, Size frameSize)
        {
            int classCount = rows - 4;

            for (int column = 0; column < columns; column++)
            {
                int bestClass = -1;
                float bestScore = 0.0f;

                for (int c = 0; c < classCount; c++)
                {
                    float score = output[0, 4 + c, column];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                if (bestScore < ConfidenceThreshold || !IsAcceptedClass(bestClass)) continue;

                float centerX = output[0, 0, column];
                float centerY = output[0, 1, column];
                float width = output[0, 2, column];
                float height = output[0, 3, column];

                AddCandidate(centerX - (width / 2.0f), centerY - (height / 2.0f), width, height,
                    bestScore, bestClass, scale, frameSize);
            }
        }

        /// <summary>YOLOv5 and v7: one row per box, with a separate objectness score.</summary>
        private void ReadLegacy(Tensor<float> output, int rows, int columns, double scale, Size frameSize)
        {
            int classCount = columns - 5;

            for (int row = 0; row < rows; row++)
            {
                float objectness = output[0, row, 4];
                if (objectness < ConfidenceThreshold) continue;

                int bestClass = -1;
                float bestScore = 0.0f;

                for (int c = 0; c < classCount; c++)
                {
                    float score = output[0, row, 5 + c];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                float confidence = objectness * bestScore;
                if (confidence < ConfidenceThreshold || !IsAcceptedClass(bestClass)) continue;

                float centerX = output[0, row, 0];
                float centerY = output[0, row, 1];
                float width = output[0, row, 2];
                float height = output[0, row, 3];

                AddCandidate(centerX - (width / 2.0f), centerY - (height / 2.0f), width, height,
                    confidence, bestClass, scale, frameSize);
            }
        }

        /// <summary>Maps a box out of the letterboxed input back onto the frame and keeps it.</summary>
        private void AddCandidate(double x, double y, double width, double height, float score,
            int classId, double scale, Size frameSize)
        {
            if (width <= 0.0 || height <= 0.0) return;

            // The frame was pasted at the top left corner of the letterbox, so undoing the scale
            // is all that is needed.
            int left = (int)Math.Round(x / scale);
            int top = (int)Math.Round(y / scale);
            int right = (int)Math.Round((x + width) / scale);
            int bottom = (int)Math.Round((y + height) / scale);

            left = Clamp(left, 0, frameSize.Width - 1);
            top = Clamp(top, 0, frameSize.Height - 1);
            right = Clamp(right, left + 1, frameSize.Width);
            bottom = Clamp(bottom, top + 1, frameSize.Height);

            candidates.Add(new Candidate(new Rectangle(left, top, right - left, bottom - top), score, classId));
        }

        /// <summary>
        /// Greedy non maximum suppression, per class. Exports that already deduplicate their
        /// output skip this.
        /// </summary>
        private void SuppressOverlaps()
        {
            candidates.Sort(delegate (Candidate left, Candidate right)
            {
                return right.Score.CompareTo(left.Score);
            });

            for (int i = 0; i < candidates.Count; i++)
            {
                if (candidates[i].Suppressed) continue;

                for (int j = i + 1; j < candidates.Count; j++)
                {
                    if (candidates[j].Suppressed) continue;
                    if (candidates[j].ClassId != candidates[i].ClassId) continue;

                    if (IntersectionOverUnion(candidates[i].Box, candidates[j].Box) > NonMaximumSuppressionThreshold)
                    {
                        candidates[j].Suppressed = true;
                    }
                }
            }
        }

        private static double IntersectionOverUnion(Rectangle left, Rectangle right)
        {
            Rectangle intersection = Rectangle.Intersect(left, right);
            if (intersection.IsEmpty) return 0.0;

            double intersectionArea = intersection.Width * (double)intersection.Height;
            double unionArea = (left.Width * (double)left.Height) +
                               (right.Width * (double)right.Height) - intersectionArea;

            return unionArea > 0.0 ? intersectionArea / unionArea : 0.0;
        }

        private bool IsAcceptedClass(int classId)
        {
            if (classId < 0) return false;
            return acceptedClasses == null || acceptedClasses.Contains(classId);
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

        private static int Clamp(int value, int min, int max)
        {
            if (max < min) return min;
            return value < min ? min : (value > max ? max : value);
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;

            session.Dispose();
            letterboxed.Dispose();
            resized.Dispose();
            emptyOutput.Dispose();
        }

        private sealed class Candidate
        {
            public readonly Rectangle Box;
            public readonly float Score;
            public readonly int ClassId;
            public bool Suppressed;

            public Candidate(Rectangle box, float score, int classId)
            {
                Box = box;
                Score = score;
                ClassId = classId;
            }
        }
    }
}
