"""Classifier registry with the hyperparameters found in the original notebooks."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config

SEED = config.SEED

DT_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
}
ET_GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
# Best parameters reported by the original grid searches.
ET_BEST = {
    "A": dict(n_estimators=150, max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features=None),
    "B": dict(n_estimators=150, max_depth=20, min_samples_split=5, min_samples_leaf=2,
              max_features="sqrt", criterion="gini", bootstrap=True),
}
XGB_BEST = dict(colsample_bytree=0.6863284109987373, gamma=0.31144523790950013,
                learning_rate=0.0356042394981304, max_depth=6, min_child_weight=3,
                n_estimators=104, subsample=0.6553412369671205)

DISPLAY = {"svm": "SVM", "dt": "Decision Tree", "rf": "Random Forest", "gb": "Gradient Boosting",
           "et": "Extra Trees", "xgb": "XGBoost", "nn": "Neural Network"}


class KerasMLP:
    """The original 3-layer MLP (512-256-128, dropout 0.5, Adam 1e-3, batch 16) on raw waveforms.

    Changes vs the original notebook:
      * the test set is scaled with the TRAINING max (original scaled test by its own max);
      * early stopping on a validation split carved from the training set (original trained a
        fixed 50 epochs, monitoring the test set): stops on val-loss stall, on success
        (val acc >= 0.995) or when stuck at chance after `chance_epochs` epochs.
    """

    def __init__(self, n_classes=config.N_TARGETS, epochs=50, batch_size=16, lr=1e-3, patience=8,
                 val_fraction=0.1, chance_epochs=10, seed=SEED, verbose=2):
        self.n_classes, self.epochs, self.batch_size, self.lr = n_classes, epochs, batch_size, lr
        self.patience, self.val_fraction, self.chance_epochs = patience, val_fraction, chance_epochs
        self.seed, self.verbose = seed, verbose
        self.classes_ = np.arange(n_classes)

    def _build(self, n_features):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed)
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(n_features,)),
            tf.keras.layers.Dense(512, activation="relu"), tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.n_classes, activation="softmax"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y):
        import tensorflow as tf
        chance = 1.0 / self.n_classes
        chance_epochs, success = self.chance_epochs, 0.995

        class StopOnSuccessOrChance(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                acc = (logs or {}).get("val_accuracy", 0.0)
                if acc >= success:
                    print(f"  early stop: success (val acc {acc:.3f})")
                    self.model.stop_training = True
                elif epoch + 1 >= chance_epochs and acc < 2 * chance:
                    print(f"  early stop: stuck near chance (val acc {acc:.3f})")
                    self.model.stop_training = True

        self.scale_ = np.max(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(X))
        n_val = int(round(self.val_fraction * len(X)))
        val, tr = perm[:n_val], perm[n_val:]
        Xs = X / self.scale_
        self.model_ = self._build(X.shape[1])
        self.history_ = self.model_.fit(
            Xs[tr], y[tr], validation_data=(Xs[val], y[val]), epochs=self.epochs,
            batch_size=self.batch_size, verbose=self.verbose,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                                        restore_best_weights=True),
                       StopOnSuccessOrChance()])
        return self

    def predict_proba(self, X):
        return self.model_.predict(X / self.scale_, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def get_model(name, pipeline, tune=False):
    """Return (estimator, input_key). input_key is 'X' (peak features) or 'X_wave' (raw waveform)."""
    if name == "svm":
        return SVC(kernel="rbf", gamma=1, C=100), "X"
    if name == "dt":   # original always grid-searched the tree (best params were not recorded)
        return GridSearchCV(DecisionTreeClassifier(random_state=SEED), DT_GRID, cv=5, n_jobs=-1), "X"
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), "X"
    if name == "gb":   # not present in the notebooks; sklearn defaults, as named in the report figures
        return GradientBoostingClassifier(random_state=SEED), "X"
    if name == "et":
        if tune:      # inside CV this becomes nested tuning (grid search on each training fold)
            return GridSearchCV(ExtraTreesClassifier(random_state=SEED, n_jobs=-1), ET_GRID,
                                cv=5, n_jobs=-1), "X"
        return ExtraTreesClassifier(**ET_BEST[pipeline], random_state=SEED, n_jobs=-1), "X"
    if name == "xgb":
        import xgboost as xgb   # optional dependency
        return xgb.XGBClassifier(**XGB_BEST, objective="multi:softprob", random_state=SEED), "X"
    if name == "nn":
        if pipeline != "B":
            raise ValueError("The original neural network was trained on pipeline-B raw waveforms; use --pipeline B")
        return KerasMLP(), "X_wave"
    raise ValueError(f"Unknown model '{name}'")
