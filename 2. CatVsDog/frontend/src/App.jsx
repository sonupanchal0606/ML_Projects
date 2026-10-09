import { useState } from "react";
import "./App.css";

function App() {
  // -------------------------------------------------------
  // State
  // -------------------------------------------------------

  // Selected image file
  const [image, setImage] = useState(null);

  // Local image preview
  const [preview, setPreview] = useState(null);

  // Backend prediction
  const [result, setResult] = useState(null);

  // Loading state
  const [loading, setLoading] = useState(false);

  // -------------------------------------------------------
  // When user selects image
  // -------------------------------------------------------

  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    // Store actual file
    setImage(file);

    // Create local browser preview
    setPreview(URL.createObjectURL(file));

    // Remove old result
    setResult(null);
  };

  // -------------------------------------------------------
  // Send image to backend
  // -------------------------------------------------------

  const predictImage = async () => {
    if (!image) {
      alert("Please select an image first.");
      return;
    }

    setLoading(true);

    // FormData is used when uploading files.
    const formData = new FormData();

    formData.append("file", image);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      setResult(data);
    } catch (error) {
      console.error(error);

      alert("Could not connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------------------------------
  // UI
  // -------------------------------------------------------

  return (
    <div className="page">
      <div className="container">
        <h1>🐱 Cat vs Dog Classifier 🐶</h1>

        <p className="subtitle">
          Upload an image and let our CNN decide whether it's a cat or dog.
        </p>

        {/* Image upload */}

        <input type="file" accept="image/*" onChange={handleImageChange} />

        {/* Image preview */}

        {preview && <img src={preview} className="preview" alt="Selected" />}

        {/* Predict button */}

        <button onClick={predictImage} disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>

        {/* Prediction */}

        {result && (
          <div className="result">
            <p>Prediction:{result.prediction}</p>

            <p>
              Confidence: <strong>{result.confidence}%</strong>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
