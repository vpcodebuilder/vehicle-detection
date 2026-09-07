# Models

Drop a YOLO `.onnx` file here and the program picks it up automatically; `--model <path>` points at
one somewhere else. Nothing here is committed - see the `.gitignore` - because the weights are
large and their licences do not necessarily match this repository.

```
curl -L -o yolov10n.onnx \
  https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx
```

YOLOv5, v8, v10 and YOLO11 are AGPL-3.0. Check that against how you intend to ship the program
before you build on one of them.

Without a model the program falls back to background subtraction, which needs no weights at all.
