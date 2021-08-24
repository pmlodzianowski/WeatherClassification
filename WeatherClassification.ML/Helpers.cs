using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WeatherClassification.ML
{
    public static class Helpers
    {
        public static string GetAbsolutePath(string relativePath)
        {
            var _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }

        public static (string trainDatasetPath, string testDatasetPath) GenerateDataSetFiles(string dataPath, uint percentOfTotalImagesForTests = 15)
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
