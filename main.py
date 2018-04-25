import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import io

############
# TRAINING #
############
trainingData = pd.read_csv('data/train.csv')
sexTransformed = []

# Replace all sexes as 0 for male, 1 for female
for row in trainingData['Sex']:
    if row == 'male':
        sexTransformed.append(0)
    elif row == 'female':
        sexTransformed.append(1)

# Create series
sexSeries = pd.Series(sexTransformed)

# Prepare training data
trainingData['sexTransformed'] = sexSeries.values
# Ensure age always has a value of at least 0
trainingData['Age'].fillna(0, inplace=True)
# Remove features
xTraining = trainingData.drop(['PassengerId','Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1).values
# Keep the survived true/false to train on
yTraining = trainingData[['Survived']].values

########
# TEST #
########


testData = pd.read_csv('data/test.csv')

sexTransformed = []

for row in testData['Sex']:
    if row == 'male':
        sexTransformed.append(0)
    elif row == 'female':
        sexTransformed.append(1)

sexSeries = pd.Series(sexTransformed)

testData['sexTransformed'] = sexSeries.values

testData['Age'].fillna(0, inplace=True)
testData['Fare'].fillna(0, inplace=True)

xTest = testData.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1).values

# Scale all data down to be in the range of 0 - 1
xScaler = MinMaxScaler(feature_range=(0, 1))
yScaler = MinMaxScaler(feature_range=(0, 1))

xScaledTraining = xScaler.fit_transform(xTraining)


yScaledTraining = yTraining

# for row in xTest:
xScaledTest = xScaler.fit_transform(xTest)
# xScaledTest = xTest

#########
# MODEL #
#########

# Learning rate
learningRate = 0.001
# How many learning epochs
learningEpochs = 10000
# When to display the current cost
displayStep = 1000

# How many inputs to the NN
numberInputs = 6
# One output, probability of survival
numberOutputs = 1


layer1Nodes = 6
layer2Nodes = 10
layer3Nodes = 4

with tf.variable_scope("input"):
    x = tf.placeholder(dtype=tf.float32, shape=(None, numberInputs))

with tf.variable_scope("layer1"):
    # weights definition
    weights = tf.get_variable(name="weights1", shape=[numberInputs, layer1Nodes], \
                              initializer=tf.contrib.layers.xavier_initializer())
    # biases definition
    biases = tf.get_variable(name="biases1", shape=[layer1Nodes], \
                             initializer=tf.zeros_initializer())
    # outputs = x matrix multiplied against weights + the biases
    layer1Outputs = tf.nn.sigmoid(tf.matmul(x, weights) + biases)

with tf.variable_scope("layer2"):
    weights = tf.get_variable(name="weights2", shape=[layer1Nodes, layer2Nodes], \
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer2Nodes], \
                             initializer=tf.zeros_initializer())
    layer2Outputs = tf.nn.sigmoid(tf.matmul(layer1Outputs, weights) + biases)

with tf.variable_scope("layer3"):
    weights = tf.get_variable(name="weights3", shape=[layer2Nodes, layer3Nodes] \
                              , initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable(name="biases3", shape=[layer3Nodes], \
                             initializer=tf.zeros_initializer())
    layer3Outputs = tf.nn.sigmoid(tf.matmul(layer2Outputs, weights) + biases)

with tf.variable_scope("output"):
    weights = tf.get_variable(name="weights4", shape=[layer3Nodes, numberOutputs] \
                              , initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[numberOutputs], \
                             initializer=tf.zeros_initializer())
    predictions = tf.nn.sigmoid(tf.matmul(layer3Outputs, weights) + biases)

############
# TRAINING #
############

with tf.variable_scope("cost"):
    # Expected value
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    cost = tf.reduce_mean(tf.squared_difference(predictions, y))

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

# Summary operation to log progress
with tf.variable_scope("logging"):
    tf.summary.scalar("currentCost", cost)
    summary = tf.summary.merge_all()

#################
# TRAINING LOOP #
#################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(learningEpochs):
        sess.run(optimizer, feed_dict={x: xScaledTraining, y: yScaledTraining})
        if epoch % displayStep == 0:
            trainingCost = sess.run(cost, feed_dict={x: xScaledTraining, y: yScaledTraining})
            print(epoch, trainingCost)

    finalTrainingCost = sess.run(cost, feed_dict={x: xScaledTraining, y: yScaledTraining})
    # finalTestCost = sess.run(cost, feed_dict = {x: xTest, y: yTest})
    predict = sess.run(predictions, feed_dict={x: xScaledTest})
    predict = sess.run(tf.round(predict))

    # print(testData['PassengerId'].values)
    results = []
    for row in predict:
        results.append(int(row[0]))
    data = {'PassengerId': testData['PassengerId'].values, 'Survived': results}

    df = pd.DataFrame(data=data)

    df.to_csv('results/results.csv', index=False)

    # print(sess.run(tf.nn.softmax(predict)))
    # print("Final Training Cost: {}", format(finalTrainingCost))
    # print("Final Testing Cost: {}", format(finalTestCost))