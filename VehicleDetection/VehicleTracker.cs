using System;
using System.Collections.Generic;
using System.Drawing;

namespace VehicleDetection
{
    /// <summary>A vehicle followed across consecutive frames.</summary>
    public sealed class TrackedVehicle
    {
        private const int MaxTrailLength = 24;

        private readonly List<Point> trail = new List<Point>(MaxTrailLength);

        public int Id { get; private set; }

        /// <summary>Bounding box seen on the most recent frame it was matched.</summary>
        public Rectangle Box { get; internal set; }

        public Point Centroid { get; internal set; }

        public Point PreviousCentroid { get; internal set; }

        /// <summary>Number of frames the track has been matched to a detection.</summary>
        public int Age { get; internal set; }

        /// <summary>Consecutive frames without a matching detection.</summary>
        public int MissingFrames { get; internal set; }

        /// <summary>True once the track has crossed the counting line.</summary>
        public bool Counted { get; internal set; }

        /// <summary>Recent centre points, oldest first.</summary>
        public IReadOnlyList<Point> Trail
        {
            get { return trail; }
        }

        internal TrackedVehicle(int id, Rectangle box)
        {
            Id = id;
            Box = box;
            Centroid = CenterOf(box);
            PreviousCentroid = Centroid;
            Age = 1;
            trail.Add(Centroid);
        }

        internal void MatchTo(Rectangle box)
        {
            Box = box;
            PreviousCentroid = Centroid;
            Centroid = CenterOf(box);
            Age++;
            MissingFrames = 0;

            trail.Add(Centroid);
            if (trail.Count > MaxTrailLength) trail.RemoveAt(0);
        }

        internal static Point CenterOf(Rectangle box)
        {
            return new Point(box.X + (box.Width / 2), box.Y + (box.Height / 2));
        }
    }

    /// <summary>
    /// Centroid tracker that gives every vehicle a stable id and counts the ones crossing a
    /// horizontal line. Without it a raw detector reports the same car again on every frame, so
    /// there is no way to say how many vehicles actually passed.
    /// </summary>
    public sealed class VehicleTracker
    {
        private readonly List<TrackedVehicle> tracks = new List<TrackedVehicle>();
        private int nextId = 1;

        /// <summary>Frames a track survives without a matching detection before it is dropped.</summary>
        public int MaxMissingFrames { get; set; }

        /// <summary>
        /// Largest accepted distance between a track and a detection, as a fraction of the frame
        /// diagonal.
        /// </summary>
        public double MaxMatchDistanceRatio { get; set; }

        /// <summary>A track has to be seen this many times before it can be counted.</summary>
        public int MinimumAgeToCount { get; set; }

        public int CountedDownwards { get; private set; }

        public int CountedUpwards { get; private set; }

        public int TotalCount
        {
            get { return CountedDownwards + CountedUpwards; }
        }

        /// <summary>Tracks that were matched on the most recent update, plus recently lost ones.</summary>
        public IReadOnlyList<TrackedVehicle> Tracks
        {
            get { return tracks; }
        }

        public VehicleTracker()
        {
            MaxMissingFrames = 8;
            MaxMatchDistanceRatio = 0.09;
            MinimumAgeToCount = 2;
        }

        public void Reset()
        {
            tracks.Clear();
            nextId = 1;
            CountedDownwards = 0;
            CountedUpwards = 0;
        }

        /// <summary>
        /// Associates the detections of the current frame with the existing tracks and updates the
        /// crossing counters.
        /// </summary>
        /// <param name="detections">Bounding boxes reported for the current frame.</param>
        /// <param name="frameSize">Size of the frame the boxes belong to.</param>
        /// <param name="countingLineY">Y coordinate of the counting line, or -1 to skip counting.</param>
        public void Update(Rectangle[] detections, Size frameSize, int countingLineY)
        {
            if (detections == null) detections = new Rectangle[0];

            double diagonal = Math.Sqrt((frameSize.Width * (double)frameSize.Width) +
                                        (frameSize.Height * (double)frameSize.Height));
            double maxDistance = diagonal * MaxMatchDistanceRatio;

            bool[] detectionUsed = new bool[detections.Length];
            bool[] trackUsed = new bool[tracks.Count];

            // Greedy nearest neighbour assignment: consider every candidate pair within range,
            // shortest distance first, and take the first assignment for each track/detection.
            List<Candidate> candidates = new List<Candidate>();
            for (int t = 0; t < tracks.Count; t++)
            {
                Point trackCenter = tracks[t].Centroid;
                for (int d = 0; d < detections.Length; d++)
                {
                    Point detectionCenter = TrackedVehicle.CenterOf(detections[d]);
                    double dx = trackCenter.X - detectionCenter.X;
                    double dy = trackCenter.Y - detectionCenter.Y;
                    double distance = Math.Sqrt((dx * dx) + (dy * dy));

                    if (distance <= maxDistance)
                    {
                        candidates.Add(new Candidate(t, d, distance));
                    }
                }
            }

            candidates.Sort(delegate (Candidate left, Candidate right)
            {
                return left.Distance.CompareTo(right.Distance);
            });

            foreach (Candidate candidate in candidates)
            {
                if (trackUsed[candidate.TrackIndex] || detectionUsed[candidate.DetectionIndex]) continue;

                trackUsed[candidate.TrackIndex] = true;
                detectionUsed[candidate.DetectionIndex] = true;
                tracks[candidate.TrackIndex].MatchTo(detections[candidate.DetectionIndex]);
            }

            // Unmatched tracks age out.
            for (int t = tracks.Count - 1; t >= 0; t--)
            {
                if (trackUsed[t]) continue;

                tracks[t].MissingFrames++;
                if (tracks[t].MissingFrames > MaxMissingFrames) tracks.RemoveAt(t);
            }

            // Unmatched detections start a new track.
            for (int d = 0; d < detections.Length; d++)
            {
                if (detectionUsed[d]) continue;
                tracks.Add(new TrackedVehicle(nextId++, detections[d]));
            }

            if (countingLineY >= 0) UpdateCounters(countingLineY);
        }

        private void UpdateCounters(int countingLineY)
        {
            foreach (TrackedVehicle track in tracks)
            {
                if (track.Counted || track.MissingFrames > 0 || track.Age < MinimumAgeToCount) continue;

                int previousY = track.PreviousCentroid.Y;
                int currentY = track.Centroid.Y;

                if (previousY < countingLineY && currentY >= countingLineY)
                {
                    track.Counted = true;
                    CountedDownwards++;
                }
                else if (previousY > countingLineY && currentY <= countingLineY)
                {
                    track.Counted = true;
                    CountedUpwards++;
                }
            }
        }

        private struct Candidate
        {
            public readonly int TrackIndex;
            public readonly int DetectionIndex;
            public readonly double Distance;

            public Candidate(int trackIndex, int detectionIndex, double distance)
            {
                TrackIndex = trackIndex;
                DetectionIndex = detectionIndex;
                Distance = distance;
            }
        }
    }
}
