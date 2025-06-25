// See https://aka.ms/new-console-template for more information

using Luxoria.Algorithm.YoLoDetectModel;

Console.Write("Creating API...");
YoLoDetectModelAPI api = new YoLoDetectModelAPI("yolov5l.onnx");

Console.Write("Detecting using API...");
var data = api.Detect("img.jpg");

Console.WriteLine(data);

Console.WriteLine($"Detected {data.Count} objects:");