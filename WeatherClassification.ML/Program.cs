using System;

namespace WeatherClassification.ML
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataPath = @"E:\repositories\private\WeatherClassification\data";
            var builderInceptionV3 = new ModelBuilder(dataPath, Microsoft.ML.Vision.ImageClassificationTrainer.Architecture.InceptionV3);
            var builderMobilenetV2 = new ModelBuilder(dataPath, Microsoft.ML.Vision.ImageClassificationTrainer.Architecture.MobilenetV2);
            var builderResnetV250 = new ModelBuilder(dataPath, Microsoft.ML.Vision.ImageClassificationTrainer.Architecture.ResnetV250);
            builderInceptionV3.CreateModel();
            builderMobilenetV2.CreateModel();
            builderResnetV250.CreateModel();
        }
    }
}
