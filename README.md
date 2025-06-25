# YoLo Detect Model
## OBB Automatic Bounding Boxes Implementation C++/CLI

This package provides a .NET wrapper for the YoLo Detect Momodel implemented in native C++ with OpenCV. It allows to gather the bounding boxes associated with class ID & its confidence score

## Requirements
- **.NET Version**: `net8.0` or compatible.
- **Native Dependencies**: OpenCV libraries are embedded within the native implementation.

## Source Code
The precompiled native libraries are built from the source code available at [LuxoriaSoft/yolo_extractor_dllcore](https://github.com/LuxoriaSoft/yolo_extractor_dllcore)

## Installation
You can install the package via NuGet Package Manager or the `.NET CLI`:

### Using NuGet Package Manager
Search for `Luxoria.Algorithm.YoLoDetectModel` in the NuGet Package Manager and install it.

### Using .NET CLI
Run the following command:
```bash
dotnet add package Luxoria.Algorithm.YoLoDetectModel --version 1.0.0
```

### Usage
```csharp	
using Luxoria.Algorithm.YoLoDetectModel;

class Program
{
    static void Main()
    {
        YoLoDetectModelAPI api = new YoLoDetectModelAPI("path_to_yolov5l.onnx");

        try
        {
            IReadOnlyCollection<Detection> data = api.Detect("path_to_img.jpg");
            Console.WriteLine($"Detected : {data.Count} objects");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}
```

### License
Luxoria.Algorithm.BrisqueScore is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

LuxoriaSoft