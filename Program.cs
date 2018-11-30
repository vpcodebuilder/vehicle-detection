using System;
using System.IO;

namespace VehicleDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                string binFolder = Directory.GetParent(Environment.CurrentDirectory).ToString();
                new Engine(binFolder + "\\Relaxing_highway_traffic.mp4").Run(new TrafficVehicleDetection()
                {
                    DetectionScale = new System.Drawing.RectangleF(0.0f, 0.5f, 1.0f, 0.5f)
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.Read();
            }
        }
    }
}
