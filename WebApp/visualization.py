import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3

class visualizer:
    def __init__(self):
        self.twitterPreds = 'TwitterPreds.csv'
        self.twitterData = 'TwitterData.csv'

    def visualizeAmAll(self):
        df = pd.read_csv('Data/amazonAll'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/amazonAll'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.Figure()
        barChartPreds(singlePred)
        return mpld3.display(fig)

    def visualizeAmFeed(self):
        df = pd.read_csv('Data/amazonFeed'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/amazonFeed'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

    def visualizeApAll(self):
        df = pd.read_csv('Data/appleAll'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/appleAll'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

    def visualizeApFeed(self):
        df = pd.read_csv('Data/appleFeed'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/appleFeed'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

    def visualizeFbAll(self):
        df = pd.read_csv('Data/facebookAll'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/facebookAll'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

    def visualizeFbFeed(self):
        df = pd.read_csv('Data/facebookFeed'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/facebookFeed'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

    def visualizeGoAll(self):
        df = pd.read_csv('Data/googleAll'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/googleAll'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

    def visualizeGoFeed(self):
        df = pd.read_csv('Data/googleFeed'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/googleFeed'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

    def visualizeNfAll(self):
        df = pd.read_csv('Data/netflixAll'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/netflixAll'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

    def visualizeNfFeed(self):
        df = pd.read_csv('Data/netflixFeed'+self.twitterData)
        tweets = df.iloc[:,[False,True,False,False,False,False,False,False,False]].to_numpy()

        df = pd.read_csv('Data/netflixFeed'+self.twitterPreds)
        predictions = df.iloc[:,[False,True,True,True,True,True]].to_numpy()

        tweets = np.array([ elem for singleList in tweets for elem in singleList])
        singlePred, confidence = interpretPreds(predictions)
        #print(singlePred)
        fig = plt.figure()
        barChartPreds(singlePred)
        # mpld3.show()
        return mpld3.display(fig)
        # return tweets, singlePred, confidence

def interpretPreds(predictions):
    singlePred = np.zeros((len(predictions)),dtype=np.intc)
    confidence = np.zeros((len(predictions)),dtype=np.float_)
    for i, pred in enumerate(predictions):
        argMaxPred = np.argmax(pred)
        conf = pred[argMaxPred]
        if(pred[argMaxPred]<0.666666):
            x = -pred[0]+pred[4]
            y = pred[2]
            neutralDist = np.sqrt(np.power((x-0),2)+np.power((y-0.666666),2))
            positiveDist = np.sqrt(np.power((x-0.666666),2)+np.power((y-0),2))
            negativeDist = np.sqrt(np.power((x-(-0.666666)),2)+np.power((y-0),2))
            minDist = np.argmin([neutralDist,positiveDist,negativeDist])
            if(minDist==positiveDist):
                mag = pred[4]
                avgPred = 4
            elif(minDist==negativeDist):
                mag = pred[0]
                avgPred = 0
            else:
                mag = pred[2]
                avgPred = 2
            confidence[i] = mag
            singlePred[i] = avgPred
        else:
            confidence[i] = conf
            singlePred[i] = argMaxPred
    return singlePred, confidence

def barChartPreds(predictions):
    negative=0
    neutral=0
    positive=0
    error = 0

    for pred in predictions:
        if(pred == 4):
            positive=positive+1
        elif(pred == 2):
            neutral=neutral+1
        elif(pred == 0):
            negative=negative+1
        else:
            error=error+1
    
    data = {
        'positive' : positive,
        'neutral' : neutral,
        'negative' : negative
    }
    names = list(data.keys())
    values = list(data.values())
    plt.title('Predictions Visualization')
    plt.bar(names, values)
    plt.ylabel('Number of tweets')
    plt.xlabel('Prediction')


def graphPlotPreds(predictions):
    xs = np.zeros((len(predictions)),dtype= np.float_)
    ys = np.zeros((len(predictions)),dtype= np.float_)
    xs = np.multiply(np.add(-predictions.T[0],predictions.T[4]),10).tolist()
    ys = np.multiply(predictions.T[2],10).tolist()
    origin=[0], [0]
    plt.quiver(*origin, xs, ys, color='r', scale=21)
    plt.title('Predictions Visualization')
    plt.ylabel('Neutral Sendtiment')
    plt.xlabel('Positive/Negative Sentiment')
    plt.ylim(-0.01,0.04)

