using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WeatherClassification.ML.Model;

namespace WeatherClassification.ML
{
    public class ModelBuilder
    {
        // Create MLContext to be shared across the model creation workflow objects 
        // Set a random seed for repeatable/deterministic results across multiple trainings.
        private MLContext _mlContext = new MLContext(seed: 1);
        private string _trainDataFilePath;
        private string _testFileDataPath;
        private string _resultsSavePath;

        private string _modelZipPath;
        private string _modelLogsPath;
        private ImageClassificationTrainer.Architecture _arch;

        public ModelBuilder(string dataPath, ImageClassificationTrainer.Architecture arch)
        {
            _arch = arch;
            (_trainDataFilePath, _testFileDataPath) = DataPreparation.GenerateDataSetFiles(dataPath);

            _modelZipPath = GetAbsolutePath($"Models/MLModel-{_arch}.zip");
            _modelLogsPath = GetAbsolutePath($"Models/ML-Model-{_arch}-Logs.txt");
        }

        public void CreateModel()
        {
            // Load Training Data
            IDataView trainingDataView = _mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: _trainDataFilePath,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Load Testing Data
            IDataView testingDataView = _mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: _testFileDataPath,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(_mlContext);

            // Train Model
            ITransformer mlModel = TrainModel(_mlContext, trainingDataView, trainingPipeline);

            // Evaluate quality of Model
            Evaluate(_mlContext, testingDataView, trainingPipeline);

            // Save model
            SaveModel(_mlContext, mlModel, _modelZipPath, trainingDataView.Schema);
        }

        #region Model Creation
        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                                      .Append(mlContext.Transforms.LoadRawImageBytes("ImageSource_featurized", null, "ImageSource"))
                                      .Append(mlContext.Transforms.CopyColumns("Features", "ImageSource_featurized"));

            // Set the training algorithm 
            var trainer = mlContext.MulticlassClassification.Trainers.ImageClassification(new ImageClassificationTrainer.Options() { LabelColumnName = "Label", FeatureColumnName = "Features" })
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============");
            return model;
        }

        private static void Evaluate(MLContext mlContext, IDataView testingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(testingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Label");
            PrintMulticlassClassificationFoldsAverageMetrics(crossValidationResults);
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }
        #endregion

        #region Helpers
        public static string GetAbsolutePath(string relativePath)
        {
            var _dataRoot = new FileInfo(typeof(ModelBuilder).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }

        public static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    MacroAccuracy = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    MicroAccuracy = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"    LogLoss for class {i + 1} = {metrics.PerClassLogLoss[i]:0.####}, the closer to 0, the better");
            }
            Console.WriteLine($"************************************************************");
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = Metrics.CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = Metrics.CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = Metrics.CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = Metrics.CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = Metrics.CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = Metrics.CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = Metrics.CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = Metrics.CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");

        }
        #endregion
    }
}
