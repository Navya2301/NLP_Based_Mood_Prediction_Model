# Emotion Detection using DistilBERT
This project aims to predict emotions from text using a fine-tuned DistilBERT model. The model is trained on the Emotion dataset from Hugging Face, which includes six emotion classes: joy, sadness, anger, fear, love, and surprise.
### Project Overview
**Objective:** Build a machine learning model that accurately predicts emotions based on text inputs.
**Dataset:** Emotion dataset from Hugging Face, consisting of text labeled with one of six emotions. Here is the link : https://huggingface.co/datasets/dair-ai/emotion
```
splits = {'train': 'split/train-00000-of-00001.parquet', 'validation': 'split/validation-00000-of-00001.parquet', 'test': 'split/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])

```

**Model:** Fine-tuned DistilBERT model with a classification head for emotion prediction.

### Methodology
1. Data Preparation: Split data into training (80%) and testing (20%).
Tokenized text using DistilBERT.
2. Training: Fine-tuned DistilBERT for 3 epochs with cross-entropy loss and AdamW optimizer.
3. Evaluation: Achieved 93.16% test accuracy, with additional evaluation using precision, recall, and F1-score.

### Results
**Test Accuracy:** 93.16%
The model generalizes well to unseen data and can be used for reliable emotion prediction.

### Usage
1. Install dependencies:
```
!pip install transformers torch scikit-learn
```
2. Training the Model:
Follow the provided notebook/script to train the model on your dataset.
3. Predicting Emotions:
Use the trained model to predict emotions for new text data by following the prediction script.
```
def test(new_text):
  inputs = tokenizer(new_text, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
  inputs = {k: v.to(device) for k, v in inputs.items()}
  model.to(device)
  model.eval()  # Set the model to evaluation mode
  with torch.no_grad():  # Disable gradient calculations
      outputs = model(**inputs)
      logits = outputs.logits
      predicted_class = torch.argmax(logits, dim=-1).item()
  emotion_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
  predicted_mood = emotion_mapping[predicted_class]
  print(f'The predicted mood is: {predicted_mood}')
```
### Example
```
test("I'm not feeling well")
```
*The predicted mood is: sadness*

### Suggestions
1. Experiment with different numbers of epochs or learning rates to optimize performance.
2. Consider fine-tuning the model on a different dataset for broader application.
3. Use early stopping to prevent overfitting.

