# Language Detection using Naive Bayes

This project demonstrates how to build a **Language Detection Model** using **Naive Bayes** and a **Bag-of-Words (CountVectorizer)** approach.  
It trains a machine learning model to classify text into different languages and saves the trained model as a pipeline for reuse.

---

## 🚀 Features
- Loads and preprocesses a dataset of text samples with their corresponding languages.  
- Cleans text (removes numbers, special characters, converts to lowercase).  
- Converts text into numeric vectors using **CountVectorizer (Bag of Words)**.  
- Trains a **Naive Bayes classifier** for language detection.  
- Evaluates performance using **Accuracy** and **F1 Score**.  
- Creates a **Pipeline** for streamlined training and prediction.  
- Saves the trained model using **Pickle** for later use.  
- Predicts the language of new unseen text.
![alt text](image.png)
---

## 📂 Project Structure
```
.
├── Language Detection.csv       # Dataset file
├── language_detection.py        # Main script (your code)
├── trained_pipeline-0.1.0.pkl   # Saved trained pipeline
└── README.md                    # Documentation
```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/language-detection.git
   cd language-detection
   ```

2. Install required Python libraries:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib
   ```

---

## 📊 Workflow

1. **Load Dataset**  
   Load `Language Detection.csv` containing text and their corresponding language labels.

2. **Preprocessing**  
   - Remove special characters, numbers, and extra symbols.  
   - Convert text to lowercase.  

3. **Encoding Labels**  
   - Convert categorical language labels into numerical values using `LabelEncoder`.  

4. **Train-Test Split**  
   - Split dataset into **80% training** and **20% testing** sets.  

5. **Vectorization (Bag of Words)**  
   - Convert text into numeric vectors using `CountVectorizer`.  

6. **Model Training**  
   - Train a **Multinomial Naive Bayes** classifier on the training data.  

7. **Evaluation**  
   - Compute **Accuracy** and **F1 Score** on test data.  

8. **Pipeline**  
   - Create a Scikit-learn pipeline with `CountVectorizer + Naive Bayes`.  
   - Save the pipeline using `pickle`.  

9. **Prediction**  
   - Load pipeline and predict language of new text (e.g., `"Ciao, come stai?"` → `"Italian"`).

---

## 🖥️ Usage

Run the Python script:
```bash
python language_detection.py
```

Example prediction inside the script:
```python
text = "Ciao, come stai?"
y = pipe.predict([text])
print("Detected Language:", le.classes_[y[0]])
```

Output:
```
Detected Language: Italian
```

---

## 📈 Evaluation Metrics
- **Accuracy**: Overall correct predictions.  
- **F1 Score**: Balanced measure of precision and recall (useful for imbalanced datasets).  

---

## 🗂️ Saved Model
The trained model pipeline is saved as:
```
trained_pipeline-0.1.0.pkl
```
You can load it later and make predictions without retraining:
```python
import pickle
with open("trained_pipeline-0.1.0.pkl", "rb") as f:
    pipe = pickle.load(f)

print(pipe.predict(["Hello, how are you?"]))  # Output: English
```

---

## ✅ Requirements
- Python 3.7+  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  

---

## 📜 License
This project is licensed under the MIT License.