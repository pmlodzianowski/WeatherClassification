using Microsoft.ML;

namespace WeatherClassification.ML.Model
{
    public class PredictionEngine
    {
        private string _modelPath;
        public PredictionEngine(string modelPath)
        {
            _modelPath = modelPath;
        }

        private PredictionEngine<ModelInput, ModelOutput> CreatePredictionEngine(string modelPath)
        {
            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
            return predEngine;
        }

        public ModelOutput Predict(ModelInput input)
        {
            using (var predEngine = CreatePredictionEngine(_modelPath))
            {
                ModelOutput result = predEngine.Predict(input);
                return result;
            }
        }
    }
}
