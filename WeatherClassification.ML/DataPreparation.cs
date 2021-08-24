using System;
using System.IO;

namespace WeatherClassification.ML
{
    public static class DataPreparation
    {
        public static (string trainDatasetPath, string testDatasetPath) GenerateDataSetFile(string dataPath, uint percentOfTotalImagesForTests = 15)
        {
            var dataSetTrainFilePath = Path.Combine(dataPath, "dataset-train.tsv");
            var dataSetTestFilePath = Path.Combine(dataPath, "dataset-test.tsv");

            using (var dataTrainFile = File.CreateText(dataSetTrainFilePath))
            using (var dataTestFile = File.CreateText(dataSetTestFilePath))
            {
                dataTrainFile.WriteLine("Label\tImageSource");
                dataTestFile.WriteLine("Label\tImageSource");

                var dirs = Directory.GetDirectories(dataPath);
                foreach (var dir in dirs)
                {
                    var files = Directory.GetFiles(dir);
                    var testSetTreshold = Math.Ceiling(files.Length * percentOfTotalImagesForTests / 100d);

                    for (int i = 0; i < files.Length; ++i)
                    {
                        var label = new DirectoryInfo(dir).Name;
                        var file = files[i];
                        var line = $"{label}\t{file}";

                        if (i < testSetTreshold)
                            dataTestFile.WriteLine(line);
                        else
                            dataTrainFile.WriteLine(line);
                    }
                }
            }

            return (dataSetTrainFilePath, dataSetTestFilePath);
        }
    }
}
