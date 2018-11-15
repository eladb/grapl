use std::collections::HashMap;

struct PredictionResults {
    scores: HashMap<String, u64>
}

impl PredictionResults {
    // Returns the best Node to expand, or None if there are no more candidates
    // for example, if we have expanded every node already
    pub fn best(&self) -> Option<Node> {
        unimplemented!()
    }
}

struct PredictionRequest {
    engagement_key: String,
}

/// Takes an engagement client and key
/// Finds all unexpanded nodes in the engagement
/// Scores each one
/// Returns PredictionResults
fn main() {

    handle_sns_sqs_json(move |request: PredictionRequest| {

    })
}


// Expansion Predictor
fn handle_prediction_request() {
    let predictions: PredictionResults = unimplemented!();
    // Expand the graph into the predicted 'best'
    emit_predictions(engagement_key, predictions);
}


fn emit_predictions(engagement_key: &str, predictions: PredictionResults) {
    unimplemented!()
}
