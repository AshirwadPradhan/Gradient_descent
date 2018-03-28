import numpy as np
import matplotlib.pyplot as plt


class SinDataset():

    def __init__(self, points=100, start=-3, stop=3, seed=6):
        self.points = points
        self.start = start
        self.stop = stop
        self.seed = seed
        self.X = np.zeros(self.points)
        self.Y = np.zeros(self.points)

    def createDataset(self):
        '''
        Create the dataset
        '''
        self.X = np.linspace(self.start, self.stop, self.points)
        np.random.seed(self.seed)
        self.Y = np.sin(self.X)+np.random.uniform(-0.5, 0.5, self.points)

        return (self.X, self.Y)

    def plotDataset(self):
        '''
        Plot the generated dataset
        '''
        plt.plot(self.X, self.Y, 'ro')
        plt.axis([np.min(self.X) - 1, np.max(self.X) + 1,
                 np.min(self.Y) - 1, np.max(self.Y) + 1])
        plt.xlabel('X points')
        plt.ylabel('Y points')
        plt.show()


class LinReg():

    def __init__(self, X, Y):
        # converts the list into list of lists
        # assign X0 = 1 and making the final list [1,x]
        self.X = np.c_[np.ones(len(X)), X]
        # assign random weigths
        self.W = np.random.uniform(size=self.X.shape[1])
        # getting the values of theta0 and theta1
        print(self.W)

        self.Y = Y
        self.Y_hat = []

    def compute_dot(self):
        '''
        Compute and return Y_hat
        y_hat = theta0 * X0 + theta1 * X1
        '''
        self.Y_hat = self.X.dot(self.W)
        return self.Y_hat

    def gradient_descent(self, alpha=0.01, epochs=100):
        '''
        Gradient descent implementation
        '''
        errlist = []
        total_exp_err = 0
        last_epoch = 0
        print('plot')
        for i in range(epochs):
            # calculate the pred using the original th0, th1
            pred_y = self.compute_dot()
            # error b/w required pred and actual data
            err = (pred_y - self.Y)**2
            # calculate the sum of all errors
            total_err = np.sum(err)

            # gradient of  the error
            grad = self.X.T.dot(err)/self.X.shape[0]

            # store error after every 100 iterations
            if i % 100 == 0:
                errlist.append(total_err)
                last_epoch += 1

            # if the new error and old error has value less
            # than 0.0005 then return
            if np.abs(total_exp_err - total_err) < 0.0005:
                g_epochs = np.linspace(0, last_epoch, len(errlist))
                plt.plot(g_epochs, errlist)
                print(g_epochs)
                print(errlist)
                plt.show()
                return errlist, last_epoch
            total_exp_err = total_err

            # update the new weights
            self.W += -alpha*grad

        print('plot')
        g_epochs = np.linspace(0, last_epoch, len(errlist))
        plt.plot(g_epochs, errlist, 'b')
        print('plot')
        plt.show()
        return errlist, last_epoch


if __name__ == '__main__':
    dataset = SinDataset()
    X, Y = dataset.createDataset()
    # dataset.plotDataset()
    linreg = LinReg(X, Y)
    print(linreg.compute_dot()[:5])
    linreg.gradient_descent(epochs=1000)
