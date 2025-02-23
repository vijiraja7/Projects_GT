

NLP project from GT

Need to find SPAM or HAM from dataset

Dataset Example:
v1	v2
ham	Go until jurong point, crazy.. Available only in bugis n great world
ham	Ok lar... Joking wif u oni...


Steps:
Step 1: Load the dataset
Step 2: Preprocessing Function
Step 3: Prepare Features and Labels
Step 4: Vectorize the text data
Step 5: Split the dataset into training and testing sets
Step 6: Train the algorithm / Model
Step 7: Make predictions
Step 8: Evaluate the model
Step 9: Test the model with new data


Combinations:

DecisionTreeClassifier			-	(Step 9: Test) Failed
LogisticRegression				-	(Step 9: Test) Failed
NLP - TfidfVectorizer + DecisionTreeClassifier	-	(Step 9: Test) Failed
NLP - TfidfVectorizer + LogisticRegression	-	(Step 9: Test) Failed

NLP - CountVectorizer + MultinomialNB		-	(Step 9: Test) Success
NLP - TfidfVectorizer + MultinomialNB		-	(Step 9: Test) Success


Lesson Learned:

1) Need to use encoding option for datasets in order to read dataset vaules properly
2) Need to search and use various vectorizer for NLP
3) Should not depend only on accuracy_score, Need to perform manual testing, At times accuracy_score might show good but resuts could be wrong
