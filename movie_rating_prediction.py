# ============================================================
# TASK 4: MOVIE RATING PREDICTION
# Collaborative Filtering using SVD
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

# MovieLens 100K dataset
# u.data contains:
# user_id, movie_id, rating, timestamp

file_path = "ml-100k/u.data"

ratings = pd.read_csv(
    file_path,
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)

print("Dataset loaded successfully!")
print()


# ------------------------------------------------------------
# 3. BASIC DATA EXPLORATION
# ------------------------------------------------------------

print("First 5 rows:")
print(ratings.head())

print("\nDataset shape:")
print(ratings.shape)

print("\nColumn names:")
print(ratings.columns.tolist())

print("\nMissing values:")
print(ratings.isnull().sum())

print("\nDuplicate rows:")
print(ratings.duplicated().sum())

print("\nRating distribution:")
print(ratings["rating"].value_counts().sort_index())


# ------------------------------------------------------------
# 4. DATA PRE-PROCESSING
# ------------------------------------------------------------

# Remove unnecessary timestamp column

ratings = ratings[
    ["userId", "movieId", "rating"]
]

# Remove missing values

ratings = ratings.dropna()

# Remove duplicate records

ratings = ratings.drop_duplicates()

print("\nDataset after preprocessing:")
print(ratings.shape)


# ------------------------------------------------------------
# 5. BASIC DATASET INFORMATION
# ------------------------------------------------------------

number_of_users = ratings["userId"].nunique()
number_of_movies = ratings["movieId"].nunique()
number_of_ratings = len(ratings)

print("\nNumber of users:", number_of_users)
print("Number of movies:", number_of_movies)
print("Number of ratings:", number_of_ratings)


# ------------------------------------------------------------
# 6. VISUALIZE RATING DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

ratings["rating"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7. PREPARE DATA FOR SURPRISE
# ------------------------------------------------------------

# MovieLens rating scale is 1 to 5

reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(
    ratings[
        ["userId", "movieId", "rating"]
    ],
    reader
)


# ------------------------------------------------------------
# 8. SPLIT DATA INTO TRAINING AND TESTING
# ------------------------------------------------------------

trainset, testset = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

print("\nTraining and testing data created.")
print("Training ratings:", trainset.n_ratings)
print("Testing ratings:", len(testset))


# ------------------------------------------------------------
# 9. BUILD SVD COLLABORATIVE FILTERING MODEL
# ------------------------------------------------------------

model = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)


# ------------------------------------------------------------
# 10. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining model...")

model.fit(trainset)

print("Model training completed!")


# ------------------------------------------------------------
# 11. MAKE PREDICTIONS
# ------------------------------------------------------------

predictions = model.test(testset)

print("\nFirst 10 predictions:")

for prediction in predictions[:10]:

    print(
        "User:", prediction.uid,
        "| Movie:", prediction.iid,
        "| Actual Rating:", prediction.r_ui,
        "| Predicted Rating:",
        round(prediction.est, 2)
    )


# ------------------------------------------------------------
# 12. EVALUATE MODEL USING RMSE
# ------------------------------------------------------------

print("\nRMSE:")

rmse = accuracy.rmse(
    predictions,
    verbose=True
)


# ------------------------------------------------------------
# 13. EVALUATE MODEL USING MAE
# ------------------------------------------------------------

print("\nMAE:")

mae = accuracy.mae(
    predictions,
    verbose=True
)


# ------------------------------------------------------------
# 14. DISPLAY FINAL PERFORMANCE
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))


# ------------------------------------------------------------
# 15. ACTUAL VS PREDICTED DATAFRAME
# ------------------------------------------------------------

results = pd.DataFrame({

    "User_ID": [
        prediction.uid
        for prediction in predictions
    ],

    "Movie_ID": [
        prediction.iid
        for prediction in predictions
    ],

    "Actual_Rating": [
        prediction.r_ui
        for prediction in predictions
    ],

    "Predicted_Rating": [
        prediction.est
        for prediction in predictions
    ]
})


print("\nActual vs Predicted Ratings:")
print(results.head(15))


# ------------------------------------------------------------
# 16. VISUALIZE ACTUAL VS PREDICTED RATINGS
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Actual_Rating"],
    results["Predicted_Rating"],
    alpha=0.3
)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

plt.title(
    "Actual vs Predicted Movie Ratings"
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 17. PREDICT A RATING FOR A SPECIFIC USER AND MOVIE
# ------------------------------------------------------------

user_id = 1
movie_id = 50

prediction = model.predict(
    user_id,
    movie_id
)

print("\n" + "=" * 50)
print("EXAMPLE MOVIE RATING PREDICTION")
print("=" * 50)

print("User ID:", user_id)
print("Movie ID:", movie_id)

print(
    "Predicted Rating:",
    round(prediction.est, 2)
)


# ------------------------------------------------------------
# 18. PREDICT TOP MOVIES FOR A USER
# ------------------------------------------------------------

user_id = 1

all_movies = ratings["movieId"].unique()

rated_movies = ratings[
    ratings["userId"] == user_id
]["movieId"].unique()

unrated_movies = [
    movie
    for movie in all_movies
    if movie not in rated_movies
]


recommendations = []

for movie_id in unrated_movies:

    prediction = model.predict(
        user_id,
        movie_id
    )

    recommendations.append(
        (
            movie_id,
            prediction.est
        )
    )


# Sort by predicted rating

recommendations = sorted(
    recommendations,
    key=lambda x: x[1],
    reverse=True
)


# Get top 10

top_10 = recommendations[:10]


print("\n" + "=" * 50)
print("TOP 10 RECOMMENDED MOVIES")
print("=" * 50)

for movie_id, predicted_rating in top_10:

    print(
        "Movie ID:",
        movie_id,
        "| Predicted Rating:",
        round(predicted_rating, 2)
    )


# ------------------------------------------------------------
# 19. SAVE PREDICTIONS
# ------------------------------------------------------------

results.to_csv(
    "movie_rating_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to "
    "'movie_rating_predictions.csv'"
)


# ------------------------------------------------------------
# 20. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 4 COMPLETED")
print("=" * 60)

print("""
Movie Rating Prediction was successfully completed.

Steps performed:

1. Loaded the MovieLens dataset
2. Explored the dataset
3. Checked missing and duplicate values
4. Pre-processed the ratings
5. Split data into training and testing sets
6. Applied SVD collaborative filtering
7. Trained the recommendation model
8. Predicted movie ratings
9. Evaluated the model using RMSE and MAE
10. Visualized actual vs predicted ratings
11. Generated movie recommendations
12. Saved prediction results
""")