# Models

`yolov10n.onnx` is committed here so the project runs out of the box. The program picks up the
first `.onnx` in this folder automatically; `--model <path>` points at one somewhere else.

## Licence

**The model has its own licence, separate from the MIT licence of this repository.**
`yolov10n.onnx` comes from
[onnx-community/yolov10n](https://huggingface.co/onnx-community/yolov10n) and, like the rest of
the Ultralytics family (YOLOv5, v8, v10, YOLO11), is **AGPL-3.0**. The MIT licence in the root of
this repository covers the source code only.

AGPL-3.0 requires that anyone you distribute the program to - including users who only reach it
over a network - can get the corresponding source. That is satisfied here because the repository
is public. If you build something closed source on top of this, replace the model: YOLOX is
Apache-2.0, and Ultralytics sell a commercial licence.

## Using a different model

```
curl -L -o yolov11n.onnx <url of the export>
```

Any other `.onnx` you drop in is ignored by git. `YoloVehicleDetection` reads the output tensor
shape and adapts to the YOLOv5/v7, YOLOv8/YOLO11 and YOLOv10 layouts, so most COCO trained exports
work without a code change.

Without any model the program falls back to background subtraction, which needs no weights at all.
