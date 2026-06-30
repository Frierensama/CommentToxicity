# Comment Toxicity Detection using Deep Learning

A Streamlit web application that detects toxic comments using a **Bidirectional LSTM (BiLSTM)** model. The application performs real-time prediction, bulk CSV prediction, and provides dataset insights and model performance metrics.

## Deployment
```bash
https://commenttoxicity.onrender.com
 ```

## Features

- Real-time toxicity prediction
- Bulk prediction using CSV upload
- Multi-label classification (6 classes)
- Data insights dashboard
- Model performance visualization
- Download prediction results

## Model

- **Architecture:** Embedding → Bidirectional LSTM → Dense
- **Framework:** TensorFlow / Keras
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

### Toxicity Classes

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Hugging Face Hub

## Project Structure

```text
├── app.py
├── utils.py
├── requirements.txt
├── tokenizer.pkl
├── train.csv
├── test.csv
├── README.md
```

# Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

## Applications

- Social Media Moderation
- Online Community Management
- Content Moderation
- Brand Safety
- E-learning Platforms
- News & Discussion Forums

