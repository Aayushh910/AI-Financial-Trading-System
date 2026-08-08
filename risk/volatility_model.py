from sklearn.ensemble import RandomForestRegressor


class VolatilityModel:

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train):

        self.model.fit(
            X_train,
            y_train
        )

    def predict(self, X_test):

        return self.model.predict(
            X_test
        )