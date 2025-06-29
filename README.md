# Garbage Classification System

An intelligent waste classification system that uses deep learning to classify waste into 6 categories: cardboard, glass, metal, paper, plastic, and trash. This project was developed as part of the AICTE initiative.

## 📋 Project Overview

This project uses a deep learning model based on MobileNetV2 to classify waste materials. It includes both training and prediction scripts, along with utilities for model evaluation.

## 🚀 Features

- **Multi-class Classification**: Classifies waste into 6 categories
- **High Accuracy**: Achieves up to 81% accuracy on the validation set
- **Easy to Use**: Simple command-line interface for predictions
- **Detailed Reports**: Generates classification reports and confusion matrices
- **Transfer Learning**: Utilizes MobileNetV2 for improved performance

## 📊 Performance Metrics

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Cardboard  | 0.88      | 0.95   | 0.92     |
| Glass      | 0.78      | 0.84   | 0.81     |
| Metal      | 0.91      | 0.82   | 0.86     |
| Paper      | 0.84      | 0.83   | 0.83     |
| Plastic    | 0.72      | 0.67   | 0.69     |
| Trash      | 0.59      | 0.63   | 0.61     |
| **Average**| **0.81**  | **0.81** | **0.81** |

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/swathy4b/Garbage-Classification-AICTE.git
   cd Garbage-Classification-AICTE
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🏋️ Training the Model

To train the model with your own dataset:

1. Place your dataset in the following structure:
   ```
   dataset/
   ├── cardboard/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   └── trash/
   ```

2. Run the training script:
   ```bash
   python training.py
   ```

   This will:
   - Train the model
   - Save the best model as `best_garbage_model.h5`
   - Generate performance visualizations

## 🔍 Making Predictions

To classify a waste image:

```bash
python prediction.py
```

When prompted, enter the path to the image you want to classify.

## 📂 Project Structure

```
Garbage-Classification-AICTE/
├── .gitignore
├── README.md
├── requirements.txt
├── training.py          # Script for training the model
├── prediction.py        # Script for making predictions
├── best_garbage_model.h5  # Pre-trained model weights
├── classification_report.txt  # Detailed performance metrics
└── archive/             # Dataset directory (not included in repo)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset used for training: [TrashNet](https://github.com/garythung/trashnet)
- Built with TensorFlow and Keras
- Part of the AICTE initiative

## 📧 Contact

For any queries or suggestions, please contact [Your Email] or open an issue on GitHub.
