import pandas as pd
import utils as ut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/test_dataset.csv')

data["Review"] = data["Review"].apply(ut.preProcess)
#print(data["Review"])

reviews = data["Review"].tolist()
reviews = [review for sublist in reviews for review in sublist]

tfidf_matrix, tfidf = ut.vectorizeText(reviews)
#print(tfidf_matrix, tfidf)

#ut.matrixToArray(tfidf_matrix, tfidf)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, data['Rating'], test_size=0.2, random_state=42)

# Convert target variable to numeric type
y_train = pd.to_numeric(y_train)
y_test = pd.to_numeric(y_test)

# Create linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Generate predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate model performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: {:.2f}".format(mse))
