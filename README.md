# Vehicle Detection

A simple C# vehicle detection from EmguCV library on console application.

## Solution

Image processing for vehicle and find by feature extraction that appear in realtime viedo stream in fast performance technical.
* Reduce the size of frame by gaussian pyramid down.
* Noise reduction processing in blur feature.
* Background subtraction by CNT algorithms (outdoor light resilient).
* Detect the vehicle aligment in partial frame or detection zone with contours processing.

## Requirement

1. Visual Studio 2017
2. Nuget Emgu.CV 3.4.3.3016 and dependency ZedGraph 5.1.7

## Using

The core of Engine class can be set the camera usb port in index (0 base index have the default value) or choose the video file play.

```C#
int cameraUSBPort = 0;
var engine = new Engine(cameraUSBPort);
```

In the video player mode can be set the video file parameter.

```C#
string videoFilename = <videoPathFilename>;
var engine = new Engine(videoFilename);
```

To run the engine use the Run() method and set the vehicle detection object class.

```C#
// Detect in full frame.
engine.Run(new TrafficVehicleDetection());

// If you want to spec in zone detection. It can be set to scale rectangle zone in 0.0-1.0 value.
// In this example we have to scale for rectangle detection in bottom half frame.
engine.Run(new TrafficVehicleDetection() {
    DetectionScale = new System.Drawing.RectangleF(0.0f, 0.5f, 1.0f, 0.5f)
});
```

To exit program use Esc key.

## Output

Find object by feature extraction.

<img src="https://drive.google.com/uc?export=view&id=1TzMkGyR2Sjg0TvVRMfHvb2U5uFAdmvMc">

Detection zone and output.

<img src="https://drive.google.com/uc?export=view&id=1Qo4yzLka5X-1M5Z_95owcKySX7U5R8IF">
