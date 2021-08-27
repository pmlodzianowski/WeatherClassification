using Microsoft.ML.Data;

namespace WeatherClassification.ML.Model
{
    public class ModelOutput
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }
        public float[] Score { get; set; }
    }
}
