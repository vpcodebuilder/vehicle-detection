# Vehicle Detection

A C# console application that detects, tracks and counts vehicles in a traffic video or a live
camera stream, built on [Emgu.CV](https://www.emgu.com/) (OpenCV) and
[ONNX Runtime](https://onnxruntime.ai/).

Two detectors are available behind one [IVehicleDetector](VehicleDetection/IVehicleDetector.cs)
interface, and both feed the same tracker, counter and display:

| | Background subtraction | YOLO |
|---|---|---|
| Needs a model file | no | yes, an `.onnx` |
| Sees a vehicle that is not moving | no | yes |
| Speed on the bundled clip (960 px) | 5 ms/frame | 70 ms/frame on a 12 core CPU |
| Tells a car from a bus | no | yes |

## Solution

**Background subtraction** ([TrafficVehicleDetection](VehicleDetection/TrafficVehicleDetection.cs)),
tuned for speed:

* Scale the frame down (gaussian pyramid, then a final resize to the exact target width).
* Blur for noise reduction.
* Background subtraction with the CNT algorithm (fast, resilient to outdoor light changes).
* Morphological open/close so one vehicle becomes one solid blob instead of a cloud of specks.
* Watershed splitting of blobs that hold several vehicles, see below.
* External contour extraction, filtered by area, aspect ratio and fill ratio.

**YOLO** ([YoloVehicleDetection](VehicleDetection/YoloVehicleDetection.cs)):

* Letterbox the frame into the square network input, keeping the aspect ratio.
* Run the network through ONNX Runtime.
* Read the boxes, keeping the COCO vehicle classes (car, motorcycle, bus, truck).
* Non maximum suppression, unless the export already deduplicates its output.

Both then share:

* An optional detection zone, so only a part of the frame is evaluated.
* Centroid tracking, which gives every vehicle a stable id and counts the ones crossing a line.

## Requirements

* .NET SDK 8.0 or newer (Visual Studio 2022 or `dotnet` on the command line).
* NuGet packages `Emgu.CV` 4.8.1.5350 and `Microsoft.ML.OnnxRuntime` 1.15.0, restored automatically.
* For the YOLO detector, one `.onnx` model in `VehicleDetection/models`. See below.

## Running

```
dotnet run --project VehicleDetection -c Release
```

The clip in `VehicleDetection/assets` is copied next to the executable on build and is used when
no source is given.

```
Usage: VehicleDetection [options]

  --detector <kind>  auto (default), yolo or bgs. auto picks yolo when a model is
                     present, otherwise background subtraction.
  --model <path>     ONNX model to use. Default: the first .onnx in models/.
  --confidence <v>   YOLO score threshold, 0.0-1.0. Default 0.35.
  --detect-every <n> Run the detector on every n-th frame only and let the tracker
                     bridge the gap. Default 1. Use 2 to make YOLO real time on a CPU.
  --video <path>     Play a video file (default: bundled Relaxing_highway_traffic.mp4).
  --camera <index>   Capture from a USB camera instead, 0 based.
  --zone x,y,w,h     Detection zone as fractions of the frame, 0.0-1.0.
                     Default 0,0.5,1,0.5 (bottom half). Use 'full' for the whole frame.
  --line <0.0-1.0>   Counting line inside the zone, or 'off'. Default 0.5.
  --width <pixels>   Processing width. Default: 640 for 4:3, 960 for wide screen.
  --strict-zone      Only count a vehicle fully inside the zone.
  --no-split         Do not split blobs that hold several vehicles.
  --no-loop          Stop at the end of the clip instead of replaying it.
  --no-mask          Hide the foreground mask window.
  -h, --help         Show this help.
```

Keys while it runs: `Esc`/`Q` quit, `Space` pause, `R` restart, `S` save a snapshot to
`snapshots/` next to the executable.

## Getting a YOLO model

No model is committed to this repository: the weights are large, and the Ultralytics family is
licensed under AGPL-3.0, which would reach into any project that ships them. Fetch one yourself
into `VehicleDetection/models`, for example:

```
curl -L -o VehicleDetection/models/yolov10n.onnx \
  https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx
```

**Check the licence of whatever model you pick against how you intend to ship the program.**
YOLOv5, v8, v10 and YOLO11 are AGPL-3.0. If that does not suit you, models such as YOLOX are
Apache-2.0 (their export needs its own decoding step, which this code does not implement yet).

`YoloVehicleDetection` reads the tensor shape and adapts to three common export layouts, so most
COCO trained exports work without a code change:

| Layout | Shape | Models |
|---|---|---|
| `EndToEnd` | `[1, boxes, 6]` | YOLOv10, already deduplicated |
| `Transposed` | `[1, 4 + classes, boxes]` | YOLOv8, YOLO11 |
| `Legacy` | `[1, boxes, 5 + classes]` | YOLOv5, YOLOv7 |

### Speed

YOLOv10n needs about 70 ms per frame on a 12 core CPU, so it cannot keep up with a 24 fps source
on its own. Measured over the first 500 frames of the bundled clip:

| | Vehicles counted | Video rate it sustains |
|---|---|---|
| `--detect-every 1` | 30 | 14 fps |
| `--detect-every 2` | 28 | **28 fps** |
| `--detect-every 3` | 23 | 42 fps |

`--detect-every 2` is the useful setting: the tracker holds the boxes across the skipped frame and
the count barely moves. At 3 the gaps get long enough to lose vehicles. For full frame rate
detection, swap `Microsoft.ML.OnnxRuntime` for `Microsoft.ML.OnnxRuntime.DirectML` (any DirectX 12
GPU) or `Microsoft.ML.OnnxRuntime.Gpu` (CUDA) and register that provider on the `SessionOptions`.

## Vehicles driving side by side

Background subtraction reports foreground, not objects. Two vehicles in neighbouring lanes that
pass the camera together touch in the mask and come out as one wide contour, so only one of them
is detected. The YOLO detector does not have this problem at all, because it recognises each
vehicle on its own. For the background subtraction path, two things address it:

**The closing kernel is anisotropic.** One vehicle breaks apart vertically in the mask (roof,
windscreen, body), while vehicles in neighbouring lanes sit next to each other horizontally. A
square 13x13 kernel applied twice closed the gap between the lanes as eagerly as the gap inside
one vehicle. The kernel is now 5x11, applied once: it still welds a vehicle together, but it no
longer bridges the lane gap.

**Blobs that still merge are cut apart** by [BlobSplitter](VehicleDetection/BlobSplitter.cs). The
distance transform of a merged blob has one local maximum per vehicle with a narrow neck between
them, which is what a marker controlled watershed needs. The maxima above `SeedRatio` of the peak
become the seeds, the watershed then cuts along the necks.

A vehicle and its own cast shadow form the same kind of blob, and splitting those apart would
invent a vehicle. Each region is therefore measured for texture (the standard deviation of its
grey values): a vehicle has windows, pillars and wheels and scores high, its shadow on the asphalt
is nearly flat and scores low. Flat regions are dropped, which also makes the surviving box tighter
than the merged blob was.

Measured over the first 1500 frames of the bundled clip, boxes at least twice as wide as they are
tall - the signature of a merge - dropped from 225 to 133, and detections rose from 2596 to 3004.
Pass `--no-split` to turn the splitter off.

## Using the API

The engine takes either a camera index or a video file, and any detector.

```csharp
int cameraUsbPort = 0;
using (var engine = new Engine(cameraUsbPort))
using (var detection = new TrafficVehicleDetection())
{
    engine.Run(detection);
}
```

```csharp
using (var engine = new Engine(videoPathFileName))
using (var detection = new YoloVehicleDetection("models/yolov10n.onnx"))
{
    engine.DetectEveryNthFrame = 2;
    engine.Run(detection);
}
```

The detection zone is a rectangle scaled to the frame, in the 0.0-1.0 range. The example below
restricts detection to the bottom half of the frame.

```csharp
using (var detection = new TrafficVehicleDetection
{
    DetectionScale = new System.Drawing.RectangleF(0.0f, 0.5f, 1.0f, 0.5f)
})
{
    engine.Run(detection);
}
```

After the run, `engine.Tracker` holds the counters:

```csharp
Console.WriteLine(engine.Tracker.TotalCount);        // vehicles that crossed the line
Console.WriteLine(engine.Tracker.CountedDownwards);  // top to bottom
Console.WriteLine(engine.Tracker.CountedUpwards);    // bottom to top
```

## Output

The `result` window shows the frame with the detection zone (blue), the counting line (yellow) and
the tracked boxes with their ids and trails (green, red once counted). The background subtraction
detector opens a second `output` window with the binary foreground mask it works on; the YOLO
detector has no such intermediate image, so that window stays closed.

## Version 2.1

* A YOLO detector through ONNX Runtime, selectable with `--detector`, alongside the original
  background subtraction. Both sit behind `IVehicleDetector`.
* `--detect-every`, so a detector slower than the source frame rate can still run in real time.

## Version 2.0

Rewritten from the 2018 version. What changed:

**Platform**

* .NET Framework 4.6.1 -> .NET 8, legacy csproj -> SDK style project with `PackageReference`.
* Emgu.CV 3.4.3.3016 -> 4.8.1.5350, so `GetCaptureProperty`/`SetCaptureProperty` became
  `Get`/`Set`. The unused ZedGraph dependency was dropped.
* The bundled clip moved from `VehicleDetection/bin` to `VehicleDetection/assets` and is copied
  to the output folder on build.

**Correctness**

* `while (CvInvoke.WaitKey() == ESC_KEY)` ended the program on *any* key press and skipped the
  cleanup on Esc. Key handling is now an explicit dispatch inside the loop.
* Frames were processed on the capture thread raised by `ImageGrabbed` while the main thread sat
  in a blocking `WaitKey()`. HighGUI is not thread safe. The loop is now synchronous.
* The video path was derived from `Environment.CurrentDirectory`, which only resolved when the
  program was launched from `bin\Debug`. It is now resolved from the executable folder.
* `resultFrame` was cloned on every frame and never disposed. Buffers are reused now, and the
  detectors and the engine are `IDisposable`.
* `Convert.ToUInt32(fourCC)` threw an `OverflowException` on backends that report no codec.
* The detection zone was computed once and cached forever, even if the frame size changed.
* The end of the video was never detected; the frame counter just kept climbing past the total.
* `throw ex` in the constructors reset the stack trace of the original exception.

**Detection quality**

* The mask was blurred and then thresholded at 0, which promoted every smeared pixel, shadows
  included, to foreground. It is now thresholded at 127 into a strictly binary mask.
* 10 erode iterations removed small vehicles entirely and shrank the surviving boxes away from
  the vehicles they belonged to. Replaced by morphological open + close.
* Contours were accepted with no size, shape or fill filtering, so noise counted as vehicles.
* A vehicle only counted if its box was *entirely* inside the detection zone, so everything
  touching the border was dropped. The default is now the centre point (`--strict-zone` restores
  the old behaviour).
* `ApproxPolyDP` was applied before `BoundingRectangle`, which only made the box less accurate.

**New**

* Splitting of merged blobs, so vehicles driving side by side in neighbouring lanes are detected
  separately instead of as one wide box.
* Centroid tracking with stable ids, trails and line crossing counts per direction.
* Command line options, pause/restart/snapshot keys, playback at the source frame rate.
* FPS measured with a `Stopwatch` on the loop instead of a `System.Timers.Timer` mutating
  counters from a thread pool thread.
