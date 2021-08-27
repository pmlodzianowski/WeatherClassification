using System;
using System.IO;
using System.Linq;
using WeatherClassification.ML.Model;

namespace WeatherClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Model path: \n");
            var modelPath = Console.ReadLine();
            Console.WriteLine("\nImages directory:\n");
            var imagesDir = Console.ReadLine();
            var files = Directory.GetFiles(imagesDir);
            var modelInputs = files.Select(x => new ModelInput() { ImageSource = x, Label = null });

            var resultFilePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), $"classification-results-{DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss")}.tsv");

            PredictionEngine prdEngine = new(modelPath);
            using (var fileWriter = File.AppendText(resultFilePath))
            {
                fileWriter.WriteLine("ImageSource\tPrediction\tScores");
                foreach (var input in modelInputs)
                {
                    var results = prdEngine.Predict(input);
                    fileWriter.WriteLine($"{input.ImageSource}\t{results.Prediction}\t{string.Join(", ", results.Score)}");
                }
            }
        }
    }
}
