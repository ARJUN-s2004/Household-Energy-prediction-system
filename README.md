# Household-Energy-prediction-system

# 🧠 AI-Powered Fitness Recommendation System

This project builds and saves multiple machine learning models to predict fitness-related metrics (such as body fat percentage or workout plans) using data from `final_dataset_BFP.csv`.

## 📂 Project Structure

📁 ai_fitness_predictor/
│
├── final_dataset_BFP.csv # Cleaned dataset
├── save_models.py # Script to train and save all models
├── lr_model.pkl # Saved Linear Regression model
├── gb_model.pkl # Saved Gradient Boosting model
├── rf_model.pkl # Saved Random Forest model
├── nn_model.pkl # Saved Neural Network (with scaler)
└── README.md # Project documentation


## 📌 Models Used

| Model                | Description                              |
|---------------------|------------------------------------------|
| Linear Regression    | Baseline linear model                    |
| Gradient Boosting    | Ensemble method for better performance  |
| Random Forest        | Robust tree-based ensemble model        |
| Neural Network       | Feed-forward NN using Scikit-learn      |

## 🛠️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-fitness-predictor.git
cd ai-fitness-predictor
