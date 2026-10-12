# Final audit patch notes

This patch preserves the existing model, training pipeline, threshold, reported metrics, UI features, and validation ranges.

## Changes made
- Fixed Flask's global exception handler so expected HTTP errors (such as 404 and 405) keep their proper status codes instead of being returned as 500 errors.
- Unexpected server errors are logged in the Flask terminal while returning a JSON error response.
- Pinned scikit-learn to 1.9.1 to match the version reported for the serialized model artifact. Install the pinned requirements in a fresh virtual environment before running the application.
- Added tests for 404 and 405 handling.

## Important limitations
- Predictions are only intended for inputs within the application's documented training-scale limits. The current model has not been validated for INR-scale Indian lending data.
- Automated tests must be run in an environment with the pinned dependencies installed. This audit environment did not have Flask installed, so the Flask test suite could not be executed here.
