// See https://aka.ms/new-console-template for more information

using Luxoria.Algorithm.YoLoDetectModel;

Console.Write("Creating API...");
YoLoDetectModelAPI api = new YoLoDetectModelAPI("yolov5l.onnx");
Console.WriteLine("Ok!");

Console.Write("Detecting using API...");
IReadOnlyCollection<Detection> data = api.Detect("img.jpg");
Console.WriteLine("Ok!");

Console.WriteLine(data);

Console.WriteLine($"Detected : {data.Count} objects");