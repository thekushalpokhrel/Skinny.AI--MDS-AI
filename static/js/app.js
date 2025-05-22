$(document).ready(function() {
    // Handle image upload and prediction
    $('#uploadForm').submit(function(e) {
        e.preventDefault();

        let file = $('#imageFile')[0].files[0];
        if (!file) return;

        let formData = new FormData();
        formData.append('file', file);

        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.error) {
                    alert(response.error);
                    return;
                }

                // Display results
                $('#previewImage').attr('src', 'static/uploads/' + response.filename);
                $('#predictionResult').text(response.prediction);
                $('#confidenceResult').text((response.confidence * 100).toFixed(2));

                // Style the alert based on confidence
                let alert = $('#resultAlert');
                alert.removeClass('alert-danger alert-warning alert-success');

                if (response.confidence > 0.75) {
                    alert.addClass('alert-success');
                } else if (response.confidence > 0.5) {
                    alert.addClass('alert-warning');
                } else {
                    alert.addClass('alert-danger');
                }

                $('#resultSection').removeClass('d-none');
            },
            error: function(xhr) {
                alert('Error: ' + xhr.responseJSON?.error || 'Prediction failed');
            }
        });
    });

    // Handle model retraining
    $('#trainForm').submit(function(e) {
        e.preventDefault();
        $('#trainingResult').html('<div class="alert alert-info">Training in progress...</div>');

        $.ajax({
            url: '/retrain',
            type: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                let html = `<div class="alert alert-success">
                    <strong>Training Complete!</strong><br>
                    Final Accuracy: ${(response.accuracy * 100).toFixed(2)}%<br>
                    Validation Accuracy: ${(response.val_accuracy * 100).toFixed(2)}%
                </div>`;
                $('#trainingResult').html(html);
            },
            error: function(xhr) {
                $('#trainingResult').html(`<div class="alert alert-danger">
                    Training failed: ${xhr.responseJSON?.message || 'Unknown error'}
                </div>`);
            }
        });
    });

    // Handle dataset upload
    $('#datasetForm').submit(function(e) {
        e.preventDefault();

        let formData = new FormData(this);
        $('#datasetResult').html('<div class="alert alert-info">Uploading images...</div>');

        $.ajax({
            url: '/add_to_dataset',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                $('#datasetResult').html(`<div class="alert alert-success">
                    Successfully added ${response.added_count} images to ${response.class_name} dataset
                </div>`);
            },
            error: function(xhr) {
                $('#datasetResult').html(`<div class="alert alert-danger">
                    Upload failed: ${xhr.responseJSON?.message || 'Unknown error'}
                </div>`);
            }
        });
    });
});

function provideFeedback(isCorrect) {
    // In a real application, you would send this feedback to the server
    // to improve the model or for analytics
    alert(`Thank you for your feedback! The diagnosis was marked as ${isCorrect ? 'correct' : 'incorrect'}.`);

    // You could implement active learning here by sending the feedback to the server
    // and potentially triggering model retraining
}