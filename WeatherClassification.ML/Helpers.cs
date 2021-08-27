using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WeatherClassification.ML
{
    public static class Helpers
    {
        public static string GetAbsolutePath(string relativePath)
        {
            var _dataRoot = new FileInfo(typeof(Helpers).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }

        public static string GenerateDataSetFiles(string dataPath)
        {
            var dataSetTrainFilePath = Path.Combine(dataPath, "dataset-train.tsv");
            using (var dataTrainFile = File.CreateText(dataSetTrainFilePath))
            {
                dataTrainFile.WriteLine("Label\tImageSource");
                var ds = GetDataSet(dataPath);
                foreach (var item in ds)
                {
                    var line = $"{item.Label}\t{item.ImageSource}";
                    dataTrainFile.WriteLine(line);
                }
            }

            return dataSetTrainFilePath;
        }

        public static List<(string Label, string ImageSource)> GetDataSet(string dataPath)
        {
            var result = new List<(string Label, string ImageSource)>();
            var dirs = Directory.GetDirectories(dataPath);
            foreach (var dir in dirs)
            {
                var label = new DirectoryInfo(dir).Name;
                var files = Directory.GetFiles(dir);
                var labeledFiles = files.Select(x => (label, x));
                result.AddRange(labeledFiles);
            }
            return result;
        }
    }
}
