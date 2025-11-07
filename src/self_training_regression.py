from copy import deepcopy
import numpy as np
import pandas as pd

class SelfTrainingRegressorCustom:
    """
    Custom self-training wrapper for regression.
    Strategy:
      - Train base_estimator on labeled training set.
      - Predict on unlabeled pool.
      - Estimate confidence per prediction (1 / (1 + std_of_ensemble)) if base is ensemble.
      - Select top-k most confident samples or all above a confidence threshold.
      - Add chosen pseudo-labeled examples to the labeled set.
      - Repeat for n_iter iterations or until no more unlabeled samples.
    Notes:
      - This implementation uses RandomForest-specific ensemble-variance trick if estimator
        has attribute `estimators_`. For other models, it falls back to no uncertainty estimate
        and will use absolute error proxy if `y_val` is provided.
    """
    def __init__(self, base_estimator, max_iter=10, add_per_iter=0.1,
                 min_samples_added=1, confidence_threshold=None, random_state=42,
                 verbose=1):
        """
        base_estimator: scikit-learn regressor (RandomForest recommended)
        max_iter: max self-training iterations
        add_per_iter: fraction of current unlabeled pool to pseudo-label each iteration (0-1)
        min_samples_added: minimum number of pseudo-labeled samples to add each iter
        confidence_threshold: if set, only add samples with confidence >= threshold
        """
        self.base = deepcopy(base_estimator)
        self.max_iter = max_iter
        self.add_per_iter = add_per_iter
        self.min_samples_added = min_samples_added
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.pseudo_label_history = []  # records number added each iter

    def _ensemble_confidence(self, fitted_model, X):
        """Estimate confidence from ensemble variance if possible.
           Returns confidence scores in [0, 1] where higher is more confident."""
        if hasattr(fitted_model, "estimators_"):
            # predict from each tree/estimator
            all_preds = np.stack([est.predict(X) for est in fitted_model.estimators_], axis=0)
            # standard deviation across estimators: higher std -> less confidence
            std = np.std(all_preds, axis=0)
            # convert to a bounded confidence score: confidence = 1 / (1 + std)
            # If your target is scaled (e.g., 0..1) you can use different transform.
            confidence = 1.0 / (1.0 + std)
            # normalize confidence to 0..1 for nicer thresholding
            # avoid division by zero
            conf_norm = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-12)
            return conf_norm
        else:
            # fallback: no ensemble -> no reliable uncertainty estimate
            # return uniform confidence so selection will be random if threshold isn't used
            return np.ones(len(X))

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """
        X_labeled, y_labeled: labeled training set (pandas DataFrame / Series or numpy arrays)
        X_unlabeled: unlabeled pool (same format)
        """
        # Convert to DataFrame/Series for concatenation ease if needed
        X_train = X_labeled.reset_index(drop=True)
        y_train = y_labeled.reset_index(drop=True)
        X_unlab = X_unlabeled.reset_index(drop=True)

        rng = np.random.RandomState(self.random_state)

        for it in range(self.max_iter):
            if self.verbose:
                print(f"\n[Self-Training] Iteration {it+1}/{self.max_iter} | Labeled size: {len(y_train)} | Unlabeled pool: {len(X_unlab)}")

            # Fit base estimator on current labeled set
            self.base.fit(X_train, y_train)

            if len(X_unlab) == 0:
                if self.verbose:
                    print("No unlabeled samples left -> stopping.")
                break

            # Predict on unlabeled
            preds = self.base.predict(X_unlab)

            # Estimate confidence
            conf = self._ensemble_confidence(self.base, X_unlab)

            # Decide how many to add
            n_to_add = int(np.ceil(self.add_per_iter * len(X_unlab)))
            n_to_add = max(n_to_add, self.min_samples_added) if len(X_unlab) > 0 else 0
            if n_to_add == 0:
                if self.verbose:
                    print("n_to_add computed as 0 -> stopping.")
                break

            # Select candidates: sort by confidence descending
            idx_sorted = np.argsort(-conf)  # indices of X_unlab by descending confidence

            # Optionally filter by threshold
            if self.confidence_threshold is not None:
                eligible = idx_sorted[conf[idx_sorted] >= self.confidence_threshold]
                selected_idx = eligible[:n_to_add]
            else:
                # take top n_to_add
                selected_idx = idx_sorted[:n_to_add]

            # If none selected (e.g., threshold too high), stop early
            if len(selected_idx) == 0:
                if self.verbose:
                    print("No samples passed the confidence threshold -> stopping.")
                break

            # Create new pseudo-labeled dataset
            X_new = X_unlab.iloc[selected_idx].reset_index(drop=True)
            y_new = pd.Series(preds[selected_idx]).reset_index(drop=True)

            # Append pseudo-labeled samples to train set
            X_train = pd.concat([X_train, X_new], ignore_index=True)
            y_train = pd.concat([y_train, y_new], ignore_index=True)

            # Remove selected samples from unlabeled pool
            mask = np.ones(len(X_unlab), dtype=bool)
            mask[selected_idx] = False
            X_unlab = X_unlab.iloc[mask].reset_index(drop=True)

            # Logging
            self.pseudo_label_history.append(len(selected_idx))
            if self.verbose:
                print(f"  -> Added {len(selected_idx)} pseudo-labeled samples (confidence mean: {conf[selected_idx].mean():.4f})")

            # If no more unlabeled, break
            if len(X_unlab) == 0:
                if self.verbose:
                    print("Unlabeled pool empty after adding pseudo-labels -> stopping.")
                break

        # final fit on the augmented labeled set
        self.base.fit(X_train, y_train)
        # store final internals
        self._final_X_train_ = X_train
        self._final_y_train_ = y_train
        self._remaining_unlabeled_ = X_unlab
        return self

    def predict(self, X):
        return self.base.predict(X)

    def get_pseudo_label_history(self):
        return self.pseudo_label_history
