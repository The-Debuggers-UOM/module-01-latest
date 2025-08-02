# Dummy class to avoid deserialization error
class ModelPerformance:
    def __init__(self, *args, **kwargs): pass

import joblib
from extract_features import extract_features_from_code

# Load model bundle
bundle = joblib.load("mccall_trained_models.joblib")
model = bundle['models']['overall_quality']

# REPLACE this with actual list from bundle['feature_columns']
feature_names = [
    'cyclomatic_complexity',
    'num_functions',
    'num_classes',
    'comment_ratio',
    'max_nesting_depth',
    'has_error_handling',
    'uses_framework',
    'num_imports',
    'avg_function_length',
    'has_recursion',
    'max_line_length',
    'is_object_oriented',
    'uses_logging'
]

# Sample Java code
sample_code = """
public class HelloWorld {
    public static void main(String[] args) {
        try {
            System.out.println("Hello, World!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
"""

language = "java"
features = extract_features_from_code(sample_code, language)

# Build input in expected order
X = [[features[name] for name in feature_names]]

# Predict
prediction = model.predict(X)[0]
print("🔍 Predicted Code Quality:", prediction)
