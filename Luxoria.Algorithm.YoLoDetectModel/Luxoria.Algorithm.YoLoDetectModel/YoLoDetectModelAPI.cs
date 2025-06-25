using System.Drawing;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Luxoria.Algorithm.YoLoDetectModel;

/// <summary>
/// Detection result record
/// Contains the coordinates of the bounding box, the opencv class ID and the confidence score (over 100)
/// </summary>
/// <param name="Box">Coordinates used to be a bounding box</param>
/// <param name="ClassId">Class ID of the object identified</param>
/// <param name="Confidence">Its confidence score</param>
public record struct Detection(Rectangle Box, int ClassId, float Confidence);

/// <summary>
/// YoLoDetectModel API
/// Invokes a C++ / P/Invoke wrapper for YOLO object detection model
/// </summary>
public class YoLoDetectModelAPI : IDisposable
{
    /// <summary>
    /// Native dynamic link library name
    /// </summary>
    private const string NativeLibraryName = "obb_extractor";
    private IntPtr _instance;
    /// <summary>
    /// No max of boxes detected
    /// </summary>
    private const int MaxBoxes = 4096;

    /// <summary>
    /// Static constructor, called at the same time of the constructor below
    /// Initializes the library by extracting, placing it in a temp directory and loading it in memory
    /// </summary>
    static YoLoDetectModelAPI()
        => ExtractAndLoadNativeLibrary();

    /// <summary>
    /// Constructor used to load the model (model can be found at : https://github.com/ultralytics/yolov5/releases/download/v7.0/)
    /// </summary>
    /// <param name="modelPath">Yolo Model X/M/L (check it out at : https://github.com/ultralytics/yolov5/releases/download/v7.0/)</param>
    /// <exception cref="FileNotFoundException">If file does`NOT exists</exception>
    /// <exception cref="InvalidOperationException">If yolo failed to initialize</exception>
    public YoLoDetectModelAPI(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found : [{modelPath}]");

        if (yolo_init(modelPath) != 0)
            throw new InvalidOperationException("Failed to initialize YOLO model");
    }

    /// <summary>
    /// Runs an object/subject detection using the YOLO model
    /// </summary>
    /// <param name="imagePath">Full path to an image file</param>
    public IReadOnlyList<Detection> Detect(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
            throw new FileNotFoundException($"Image file not found : {imagePath}");

        var nativeBoxes = new YoloBBox[MaxBoxes];

        int count = yolo_detect(imagePath, nativeBoxes, nativeBoxes.Length);
        if (count < 0)
            throw new InvalidOperationException($"Native detection failed with code {count}");

        var results = new List<Detection>(count);
        for (int i = 0; i < count; i++)
        {
            var b = nativeBoxes[i];
            results.Add(new Detection(
                new Rectangle(b.x, b.y, b.w, b.h),
                b.classId,
                b.confidence));
        }
        return results;
    }

    /// <summary>
    /// Disposes the model after usage
    /// Cleans up the memory and frees allocated resources
    /// </summary>
    public void Dispose()
        => yolo_free();

    #region P/Invoke DLL Loader System

    /// <summary>
    /// Struct used to store information about a bounding box
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct YoloBBox
    {
        public int x, y, w, h;
        public int classId;
        public float confidence;
    }

    /// <summary>
    /// Initializes the model using the path
    /// </summary>
    /// <param name="onnxPath">Path to the model (onnx format)</param>
    /// <returns></returns>
    [DllImport(NativeLibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int yolo_init(string onnxPath);

    /// <summary>
    /// Calls the detection function
    /// </summary>
    /// <param name="imagePath">Path to the image</param>
    /// <param name="outBoxes">Out (result)</param>
    /// <param name="maxBoxes">Max boxes (default 4096)</param>
    /// <returns></returns>

    [DllImport(NativeLibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int yolo_detect(string imagePath, [Out] YoloBBox[] outBoxes, int maxBoxes);

    [DllImport(NativeLibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void yolo_free();

    /// <summary>
    /// Extracts and loads the library from embedded resources depending on the Runtime arch
    /// </summary>
    /// <exception cref="NotSupportedException">Architecture not being supported</exception>
    /// <exception cref="FileNotFoundException">Cannot find the specified file</exception>
    private static void ExtractAndLoadNativeLibrary()
    {
        string architecture = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X86 => "x86",
            Architecture.X64 => "x64",
            Architecture.Arm64 => "arm64",
            _ => throw new NotSupportedException("Unsupported architecture")
        };

        string resourceName = $"Luxoria.Algorithm.YoLoDetectModel.NativeLibraries.{architecture}.{NativeLibraryName}.dll";

        string tempPath = Path.Combine(Path.GetTempPath(), "LuxoriaNative");
        Directory.CreateDirectory(tempPath);

        string dllPath = Path.Combine(tempPath, $"{NativeLibraryName}.dll");

        // Extract the DLL from embedded resources
        using (Stream? resourceStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName))
        {
            if (resourceStream == null)
                throw new FileNotFoundException($"Embedded resource not found: {resourceName}");

            using (FileStream fileStream = new FileStream(dllPath, FileMode.Create, FileAccess.Write))
            {
                resourceStream.CopyTo(fileStream);
            }
        }

        // Load the extracted native library
        NativeLibrary.Load(dllPath);
    }


    #endregion
}
