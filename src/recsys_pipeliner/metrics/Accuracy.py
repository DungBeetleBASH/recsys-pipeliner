from sklearn.metrics import root_mean_squared_error, mean_absolute_error

class Accuracy:

    @staticmethod
    def rmse(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)