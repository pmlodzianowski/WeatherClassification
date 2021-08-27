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

        private string _modelZipPath;
        private string _modelLogsPath;
        private ImageClassificationTrainer.Architecture _arch;

        public ModelBuilder(string dataPath, ImageClassificationTrainer.Architecture arch)
        {
            _trainDataFilePath = Helpers.GenerateDataSetFiles(dataPath);
            _arch = arch;

            var modelDirectory = Helpers.GetAbsolutePath($"Models");
            Directory.CreateDirectory(modelDirectory);

            _modelZipPath = Helpers.GetAbsolutePath(Path.Combine(modelDirectory, $"MLModel-{_arch}.zip"));
            _modelLogsPath = Helpers.GetAbsolutePath(Path.Combine(modelDirectory, $"ML-Model-{_arch}-Logs.txt"));

            if (File.Exists(_modelZipPath))
                File.Delete(_modelZipPath);

            if (File.Exists(_modelLogsPath))
                File.Delete(_modelLogsPath);
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

            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(_mlContext);

            // Train Model
            ITransformer mlModel = TrainModel(trainingDataView, trainingPipeline);

            // Evaluate quality of Model
            Evaluate(_mlContext, trainingDataView, trainingPipeline);

            // Save model
            SaveModel(_mlContext, mlModel, trainingDataView.Schema);
        }

        #region Model Creation
        private IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                                      .Append(mlContext.Transforms.LoadRawImageBytes("ImageSource_featurized", null, "ImageSource"))
                                      .Append(mlContext.Transforms.CopyColumns("Features", "ImageSource_featurized"));

            // Set the training algorithm 
            var trainer = mlContext.MulticlassClassification.Trainers.ImageClassification(new ImageClassificationTrainer.Options() { LabelColumnName = "Label", FeatureColumnName = "Features", Arch = _arch })
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);
            return trainingPipeline;
        }

        private ITransformer TrainModel(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Log(Environment.NewLine);
            Log($"=============== Training  model (architecture: {_arch}) ===============");
            ITransformer model = trainingPipeline.Fit(trainingDataView);
            Log("=============== End of training process ===============");
            return model;
        }

        private void Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Log("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 10, labelColumnName: "Label");
            PrintMulticlassClassificationFoldsAverageMetrics(crossValidationResults);
            foreach (var result in crossValidationResults)
            {
                Log($"Fold #{result.Fold} details");
                PrintMulticlassClassificationMetrics(result.Metrics);
            }
        }

        private void SaveModel(MLContext mlContext, ITransformer mlModel, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, _modelZipPath);
            Console.WriteLine("The model is saved to {0}", _modelZipPath);
        }
        #endregion

        #region Helpers
        private void Log(string message)
        {
            using (var logFileWriter = File.AppendText(_modelLogsPath))
            {
                logFileWriter.WriteLine($"[{DateTime.UtcNow.ToString("u")}] | {message}");
            }
        }

        public void PrintConfusionMatrix(ConfusionMatrix confusionMatrix)
        {
            Log(confusionMatrix.GetFormattedConfusionTable());
        }

        public void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics)
        {
            Log($"************************************************************");
            Log($"*    Metrics for multi-class classification model   ");
            Log($"*-----------------------------------------------------------");
            Log($"    MacroAccuracy = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Log($"    MicroAccuracy = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Log($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Log($"    LogLoss for class {i + 1} = {metrics.PerClassLogLoss[i]:0.####}, the closer to 0, the better");
            }
            Log($"************************************************************");

            PrintConfusionMatrix(metrics.ConfusionMatrix);
        }

        public void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
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

            Log($"*************************************************************************************************************");
            Log($"*       Metrics for Multi-class Classification model      ");
            Log($"*------------------------------------------------------------------------------------------------------------");
            Log($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Log($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Log($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Log($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Log($"*************************************************************************************************************");
        }
        #endregion
    }
}
