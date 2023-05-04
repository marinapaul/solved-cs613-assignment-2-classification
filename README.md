Download Link: https://assignmentchef.com/product/solved-cs613-assignment-2-classification
<br>
5/5 - (1 vote)




In this assignment you will perform classification using Logistic Regression, Naive Bayes and Decision Tree classifiers. You will run your implementations on a binary class dataset and report your results.

You may not use any functions from a ML library in your code unless explicitly told otherwise.

1

Datasets

Iris Dataset (sklearn.datasets.load iris) The Iris flower data set or Fishers Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.The iris data set is widely used as a beginner’s dataset for machine learning purposes. The dataset is included in the machine learning package Scikit-learn, so that users can access it without having to find a source for it. The following python code illustrates usage.

from sklearn . datasets import load iris iris = load iris()

Spambase Dataset (spambase.data) This dataset consists of 4601 instances of data, each with 57 features and a class label designating if the sample is spam or not. The features are real valued and are described in much detail here:

https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names Data obtained from: https://archive.ics.uci.edu/ml/datasets/Spambase

2

<table>

 <tbody>

  <tr>

   <td></td>

   <td></td>

   <td></td>

   <td></td>

  </tr>

  <tr>

   <td></td>

   <td></td>

   <td></td>

   <td></td>

  </tr>

 </tbody>

</table>

<table>

 <tbody>

  <tr>

   <td></td>

   <td></td>

   <td></td>

  </tr>

  <tr>

   <td></td>

   <td></td>

   <td></td>

  </tr>

 </tbody>

</table>

1 Theory

1. Consider the following set of training examples for an unknown target function: (x1, x2) → y:

Y

x1

x2

Count

+ + + + – – – –

T T F F T T F F

T F T F T F T F

3 4 4 1 0 1 3 5

<ol>

 <li>(a)  What is the sample entropy, H(Y ) from this training data (using log base 2) (2pts)?</li>

 <li>(b)  What are the information gains for branching on variables x1 and x2 (2pts)?</li>

 <li>(c)  Draw the deicion tree that would be learned by the ID3 algorithm without pruning from this training data (3pts)?</li>

</ol>

2. We decided that maybe we can use the number of characters and the average word length an essay to determine if the student should get an A in a class or not. Below are five samples of this data:

# of Chars

Average Word Length

Give an A

<pre>2166930260393</pre>

5.68 4.78 2.31 3.16 4.2

Yes Yes No Yes No

<ol>

 <li>(a)  Whataretheclasspriors,P(A=Yes),P(A=No)? (2pt)</li>

 <li>(b)  Find the parameters of the Gaussians necessary to do Gaussian Naive Bayes classification on this decision to give an A or not. Standardize the features first over all the data together so that there is no unfair bias towards the features of different scales (2pts).</li>

 <li>(c)  Using your response from the prior question, determine if an essay with 242 characters and an average word length of 4.56 should get an A or not (3pts).</li>

</ol>

3. Consider the following questions pertaining to a k-Nearest Neighbors algorithm (1pt): (a) How could you use a validation set to determine the user-defined parameter k?

3

2 Logistic Regression

Let’s train and test a Logistic Regression Classifier to classify flowers from the Iris Dataset.

First download import the data from sklearn.datasets. As mentioned in the Datasets area, The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. We will map this into a binary classification problem between Iris setosa versus Iris virgincia and versicolor. We will use just the first 2 features, width and length of the sepals.

For this part, we will be practicing gradient descent with logistic regression. Use the following code to load the data, and binarize the target values.

iris = skdata.load iris() X= iris.data[:, :2]

y = (iris.target != 0) ∗ 1

Write a script that:

<ol>

 <li>Reads in the data with the script above.</li>

 <li>Standardizes the data using the mean and standard deviation</li>

 <li>Initialize the parameters of θ using random values in the range [-1, 1]</li>

 <li>Do batch gradient descent</li>

 <li>Terminate when absolute value change in the loss on the data is less than 2−23, or after 10, 000 iterations have passed (whichever occurs first).</li>

 <li>Use a learning rate η = 0.01.</li>

 <li>While the termination criteria (mentioned above in the implementation details) hasn’t beenmet(a) Compute the loss of the data using the logistic regression cost(b) Update each parameter using batch gradient descent</li>

</ol>

Plot the data and the decision boundary using matplotlib. Verify your solution with the Logisti-

cRegression sklearn method.

from sklearn . linear model import LogisticRegressionlgr = LogisticRegression(penalty=’none’,solver=’lbfgs ’,max iter=10000) lgr . fit (X,y)

In your writeup, present the thetas from gradient descent that minimize the loss function as well as plots of your method versus the built in LogisticRegression method.

4

3 Logistic Regression Spam Classification

Let’s train and test a Logistic Regression Classifier to classifiy Spam or Not from the Spambase Dataset.

First download the dataset spambase.data from Blackboard. As mentioned in the Datasets area, this dataset contains 4601 rows of data, each with 57 continuous valued features followed by a binary class label (0=not-spam, 1=spam). There is no header information in this file and the data is comma separated.

Write a script that:

<ol>

 <li>Reads in the data.</li>

 <li>Randomizes the data.</li>

 <li>Selects the first 2/3 (round up) of the data for training and the remaining for testing (you may use sklearn train test split for this part)</li>

 <li>Standardizes the data (except for the last column of course) using the training data</li>

 <li>Initialize the parameters of θ using random values in the range [-1, 1]</li>

 <li>Do batch gradient descent</li>

 <li>Terminate when absolute value change in the loss on the data is less than 2−23, or after 1,500 iterations have passed (whichever occurs first, this will likely be a slow process).</li>

 <li>Use a learning rate η = 0.01.</li>

 <li>Classify each testing sample using the model and choosing the class label based on which classprobability is higher.</li>

 <li>Computes the following statistics using the testing data results:(a) Precision (b) Recall(c) F-measure (d) Accuracy</li>

</ol>

Implementation Details

1. Seed the random number generate with zero prior to randomizing the data 2. There are a lot of θs and this will likely be a slow process

In your report you will need:

1. The statistics requested for your Logistic classifier run.

5

4 Naive Bayes Classifier

Let’s train and test a Naive Bayes Classifier to classifiy Spam or Not from the Spambase Dataset.

First download the dataset spambase.data from Blackboard. As mentioned in the Datasets area, this dataset contains 4601 rows of data, each with 57 continuous valued features followed by a binary class label (0=not-spam, 1=spam). There is no header information in this file and the data is comma separated. As always, your code should work on any dataset that lacks header information and has several comma-separated continuous-valued features followed by a class id ∈ 0, 1.

Write a script that:

<ol>

 <li>Reads in the data.</li>

 <li>Randomizes the data.</li>

 <li>Selects the first 2/3 (round up) of the data for training and the remaining for testing</li>

 <li>Standardizes the data (except for the last column of course) using the training data</li>

 <li>Divides the training data into two groups: Spam samples, Non-Spam samples.</li>

 <li>Creates Normal models for each feature for each class.</li>

 <li>Classify each testing sample using these models and choosing the class label based on which class probability is higher.</li>

 <li>Computes the following statistics using the testing data results:(a) Precision (b) Recall(c) F-measure (d) Accuracy</li>

</ol>

Implementation Details

<ol>

 <li>Seed the random number generate with zero prior to randomizing the data</li>

 <li>You may want to consider using the log-exponent trick to avoid underflow issues. Here’s a link about it: https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp- trick-works-in-naive-bayes</li>

 <li>If you decide to work in log space, realize that python interprets 0log0 as inf. You should identify this situation and either add an EPS (very small positive number) or consider it to be a value of zero.</li>

</ol>

In your report you will need:

1. The statistics requested for your Naive Bayes classifier run. 6

5 Decision Trees

Let’s train and test a Decision Tree to classify Spam or Not from the Spambase Dataset.

Write a script that:

1. Reads in the data.2. Randomizes the data.3. Selects the first 2/3 (round up) of the data for training and the remaining for testing 4. Standardizes the data (except for the last column of course) using the training data 5. Divides the training data into two groups: Spam samples, Non-Spam samples.6. Trains a decision tree using the ID3 algorithm without any pruning.7. Classify each testing sample using your trained decision tree.8. Computes the following statistics using the testing data results:

(a) Precision (b) Recall

(c) F-measure (d) Accuracy

Implementation Details

<ol>

 <li>Seed the random number generate with zero prior to randomizing the data</li>

 <li>Depending on your perspective, the features are either continuous or finite discretize. The latter can be considered tru since the real-values are just the number of times a feature is observed in an email, normalized by some other count. That being said, for a decision tree we normally use categorical or discretized features. So for the purpose of this dataset, look at the range of each feature and turn them into binary features by choosing a threshold. I suggest using the median or mean.</li>

</ol>

In your report you will need:

1. The statistics requested for your Decision Tree classifier run.

7

Submission

For your submission, upload to Blackboard a single zip file containing:

1. PDF Writeup2. Python notebook Code

The PDF document should contain the following:

1. Part 1:(a) Answers to Theory Questions

2. Part 2:(a) Requested Logistic Regression thetas and plots

3. Part 3:(a) Requested Classification Statistics

4. Part 4:(a) Requested Classification Statistics

5. Part 5:(a) Requested Classification Statistics