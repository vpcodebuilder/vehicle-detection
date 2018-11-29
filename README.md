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

To run the engine use the Run() method.

```C#
engine.Run();
```

To exit program use Esc key.

## Output
