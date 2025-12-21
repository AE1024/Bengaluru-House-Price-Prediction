# 🏘️ Bengaluru House Price Prediction

This project focuses on building a machine learning model to predict real estate prices in Bengaluru, India. The primary challenge was handling a messy dataset with inconsistent features and significant outliers.

---

## 🛠️ The Data Science Workflow

Instead of just feeding raw data into a model, I performed extensive data engineering to ensure high accuracy.

### 1. Data Cleaning 🧹
* **Feature Selection:** Dropped columns that didn't contribute much to price prediction, such as `area_type`, `availability`, and `society`.
* **Handling Missing Values:** Cleaned records with null values in critical features like `size` and `bath`.
* **Standardizing "BHK":** The `size` column had inconsistent strings (e.g., "2 BHK" vs "2 Bedroom"). I extracted the numeric values to create a clean `bhk` feature.
* **Processing Square Footage:** Some entries in `total_sqft` were ranges (e.g., "2100 - 2850"). I wrote a function to convert these into a single average numeric value.

### 2. Feature Engineering ✨
* **Dimensionality Reduction:** There were over 1,300 unique locations. To prevent the "curse of dimensionality," I tagged any location with fewer than 10 data points as **"other"**.
* **Price Per Sqft:** Created a `price_per_sqft` feature to help identify and remove price outliers that didn't make sense.

### 3. Outlier Removal (The Secret Sauce) 📉
To make the model robust, I removed data points that violated real-world logic:
* **BHK vs Size:** Removed houses where the square footage per bedroom was less than 300 (e.g., a 600 sqft house with 6 bedrooms).
* **Price Anomalies:** Used **Mean and Standard Deviation** per location to filter out houses that were unusually cheap or expensive for their specific area.
* **Consistency Check:** Removed records where a 2 BHK house was more expensive than a 3 BHK house in the same location with similar square footage.

---

## 🤖 Model Selection & Results

I used **GridSearchCV** to compare different algorithms and find the best hyperparameters.

| Algorithm | $R^2$ Score |
| :--- | :---: |
| **Linear Regression** | **84.7%** |
| **Decision Tree** | 72.2% |
| **Lasso Regression** | 69.8% |

The **Linear Regression** model performed best, achieving an accuracy of approximately **85%** after the rigorous cleaning process.

---

## 📂 Project Structure
* `analysis.ipynb`: The full Python notebook containing the analysis.
* `Bengaluru_House_Data.csv`: The raw dataset used for training.
* `house_price.pkl`: The final trained model saved for production.
* `columns.json`: A list of all location features for easy integration with a web app.

---

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas numpy matplotlib scikit-learn`.
3. Run the `analysis.ipynb` notebook to see the step-by-step transformation.
