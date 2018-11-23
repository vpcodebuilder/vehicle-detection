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
                new Engine(binFolder + "\\Relaxing_highway_traffic.mp4").Run();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.Read();
            }
        }
    }
}
