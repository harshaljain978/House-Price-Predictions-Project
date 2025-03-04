# House Price Predictions Project Report

Hi! I’m excited to share my project on predicting house prices using a dataset called `housing.csv`. I built this project in Google Colab, and in this report, I’ll walk you through each part of my code, explaining what I did and why, step by step. My goal was to understand the data, create some visuals, and use different machine learning models to predict house prices. Let’s dive in!

---

## Section 1: Setting Up the Tools (Import Libraries)
In the first part, I brought in all the tools I’d need for my project. These tools (called libraries in Python) help me work with data, make graphs, and build prediction models. Here’s what I used:

- `matplotlib` and `seaborn` for making charts and graphs.
- `numpy` and `pandas` for handling numbers and data tables.
- `warnings` to stop annoying warning messages from popping up.
- A bunch of tools from `sklearn` (like `LinearRegression`, `RandomForestRegressor`, etc.) to create prediction models and check how good they are.
- `%matplotlib inline` to make sure my graphs show up right in Colab.

This step is like gathering my toolbox before starting work—it makes sure everything I need is ready!

---

## Section 2: Loading the Data and First Look
Next, I loaded my dataset (`housing.csv`) into the project. Since I was using Google Colab, I first uploaded the file using a special command:

```python
from google.colab import files
uploaded = files.upload()
```

After uploading, I read the file into a table (called a DataFrame) using `pandas`. Then, I took a quick peek at it:

- I showed 5 random rows with `df.sample(5)` to see what the data looks like.
- I counted the columns with `len(df.columns)` (there’s 13 of them!) and printed their names to know what information I’m working with—like `price`, `area`, `bedrooms`, etc.

This part was like opening a book and flipping through a few pages to get a feel for it.

---

## Section 3: Digging Deeper into the Data
Now, I wanted to learn more about my data:

- `df.info()` told me the types of data (like numbers or words) and if anything was missing. Good news—no missing values!
- `df.describe()` gave me stats like the average price, smallest area, biggest number of bedrooms, etc. This helped me understand the range of my data.
- `df.isnull().sum()` confirmed there were no empty spots in the data.
- `df.nunique()` showed how many unique values each column has—like `furnishingstatus` has 3 options (furnished, semi-furnished, unfurnished).

This step was like doing a health check on my data to make sure it’s ready to use.

---

## Section 4: Checking House Prices (Price Distribution)
I wanted to see how house prices are spread out, so I made a graph:

- I used `sns.histplot` to draw a histogram of `price` with a smooth line (kde=True) to show the pattern.
- I added a title "Distribution of House Prices" to make it clear.

The graph showed that most houses cost less, but a few are super expensive. This was my first real look at the prices I’d be predicting!

---

## Section 5: Exploring Number Data (Numerical Features Distribution)
My dataset has numbers like `area`, `bedrooms`, and `parking`, so I checked how they’re distributed:

- I picked out all number columns with `df.select_dtypes(['int64', 'float64'])`.
- For each one (except `price`), I made a histogram with a smooth line to see their patterns.

For example, I saw most houses have 3-4 bedrooms, and areas vary a lot. This helped me understand the numbers behind the houses.

---

## Section 6: Looking at Word Data (Categorical Features Visualization)
Some columns have words (like `yes` or `no` for `mainroad`), so I made bar graphs for them:

- I found these columns with `df.select_dtypes(['object'])`.
- For each one, I used `sns.countplot` to count how many times each word appears (like how many houses have a guestroom).

I rotated the labels with `plt.xticks(rotation=45)` so they’re easy to read. This showed me things like most houses are on a main road and few have hot water heating.

---

## Section 7: Prices vs Numbers (Scatter Plots)
I wanted to see if things like `area` or `bedrooms` affect `price`, so I made scatter plots:

- For each number column (except `price`), I plotted it against `price` with `sns.scatterplot`.
- I labeled the axes and gave each a title like "Area VS Price".

The plots showed bigger areas usually mean higher prices, but bedrooms didn’t show as clear a pattern. This was cool to explore!

---

## Section 8: Finding Connections (Correlation Analysis)
Next, I checked how strongly the numbers relate to `price`:

- I made a correlation matrix with `df[numerical_features].corr()` to see how everything connects.
- I focused on columns where the connection to `price` was strong (more than 0.3) and made a heatmap with `sns.heatmap`.
- The heatmap uses colors (red for strong, blue for weak) and numbers to show these links.

I found `area`, `bedrooms`, `bathrooms`, `stories`, and `parking` have decent connections to `price`. I printed these names to keep track of them.

---

## Section 9: Getting Ready to Predict (Data Preparation)
Now, I set up my data for making predictions:

- My target (what I want to predict) is `price`.
- I chose features (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`) that connect well with `price`.
- I split the data into training (80%) and testing (20%) sets with `train_test_split` so I could teach my models and then test them.

I printed the sizes (436 for training, 109 for testing) to confirm it worked.

---

## Section 10: Building and Testing Prediction Models
Here’s where I built my prediction models:

- I picked 5 models: Linear Regression, Decision Tree, Random Forest, SVR, and KNN.
- For each one:
  - I trained it with `model.fit` using the training data.
  - I made predictions with `model.predict` on the test data.
  - I checked how good they were with 4 measures: MAE (average error), MSE (squared error), RMSE (root of MSE), and R² (how well it fits, 1 is perfect).

I stored all these results in a dictionary called `results`.

---

## Section 11: Showing the Results
Finally, I showed how my models did:

- I printed each model’s name and its scores (MAE, MSE, RMSE, R²).
- I made a table with `pd.DataFrame(results).T` to compare them side by side.

Here’s what I found:
- **Random Forest** was the best (lowest errors, R² of 0.58), meaning it predicted prices pretty well.
- **Linear Regression** was okay (R² of 0.50).
- **SVR** was terrible (negative R², so it’s worse than guessing!).
- Decision Tree and KNN were in the middle.

The table made it easy to see Random Forest won!

---

## What I Learned
This project was awesome! I learned how to:
- Explore data with tables and graphs.
- Spot patterns between things like area and price.
- Use different models to predict prices and pick the best one.

Random Forest worked best, but I could make it even better by tweaking it or adding the word columns (like `mainroad`) after turning them into numbers. I’m proud of this project and excited to keep learning!
