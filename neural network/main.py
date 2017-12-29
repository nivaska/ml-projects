from network import *

print("\nLoading Iris train data ")
testDataPath = "irisTestData.txt"
testDataset = IrisDataset(testDataPath)
trainDataset = IrisDataset("irisTrainData.txt")

net = BasicNeuralNetwork()
net.train(trainDataset,testDataset)


#net.load('./task_2a_new.save')
acc = net.accuracy(testDataset)
if acc > 0.5:
    print('Forwarding seems to function correctly')
else:
    print('There seem to be errors with your computation of the network outputs')
#exit()

