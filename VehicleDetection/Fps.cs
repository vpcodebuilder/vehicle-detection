using System.Timers;

namespace VehicleDetection
{
    public class Fps
    {
        private Timer secondTimer;

        public int FrameCounter
        {
            get;
            private set;
        }

        public double AverageFrameRate
        {
            get;
            private set;
        }

        public Fps(double initialFrameRate)
        {
            secondTimer = new Timer(1000);
            secondTimer.Elapsed += SecondTimer_Elapsed;
            AverageFrameRate = initialFrameRate;
        }

        public void Start()
        {
            FrameCounter = 0;
            secondTimer.Start();
        }

        public void Stop()
        {
            secondTimer.Stop();
        }

        public void UpdateFrameCount()
        {
            FrameCounter++;
        }

        private void SecondTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            AverageFrameRate = (AverageFrameRate + FrameCounter) / 2.0;
            FrameCounter = 0;
        }
    }
}