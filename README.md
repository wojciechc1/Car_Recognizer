main problems: 
 - used to much RAM. - couldn't deploy on streamlit
 - couln't find good datasets (cars pics and models info) - The main idea was to extract features by ml and then match it with specific model. 


# 🚗 Car Image Classification Pipeline

This system analyzes car images and classifies them step-by-step based on the detected view of the vehicle.

---

## 🧠 System Overview

### 1. View Detection

The pipeline starts by classifying the car's view using `ViewClassifier`.  
Possible outputs are:

- `front`
- `side`
- `rear`

---

### 2. Processing Based on View

Each detected view triggers a specific set of models:

#### If view is `front`:

- ➤ `ContextClassifier` – Detects car brand (e.g., Toyota, Audi)  
- ➤ `LogoDetector` – Detects logo location and label  

#### If view is `side`:

- ➤ `CarTypeClassifier` – Classifies car body type (e.g., SUV, sedan)

#### If view is `rear`:

- ➤ `LogoDetector`  
- ➤ `ModelDetector` – Identifies the specific model (e.g., Corolla, A4)

---

### 3. Output

The system saves the result in a structured `JSON` format containing:

```json
{
  "view": {
    "label_idx": 0,
    "label_name": "front",
    "probabilities": [ ... ]
  },
  "context": {
    "label_idx": 1,
    "label_name": "toyota",
    "probabilities": [ ... ]
  },
  "logo": [
    {
      "label": "toyota",
      "confidence": 0.68,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
