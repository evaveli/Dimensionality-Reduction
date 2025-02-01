from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def simple_imputation(df, continuous_vars):
    df = df.copy()

    for var in continuous_vars:
        print(f"Processing variable: {var}")

        # Missing mask
        missing_mask = df[var].isnull()

        missing_count = missing_mask.sum()

        if missing_count == 0:
            print(f"No missing values in {var}. Skipping.")
            continue
        else:
            print(f"Missing values in {var}: {missing_count}")

        # Simple mean imputation for continuous variables
        mean_value = df[var].mean()

        # Impute missing values
        df.loc[missing_mask, var] = mean_value

        print(f"Imputed missing values for {var} using mean value: {mean_value:.4f}")

    return df


def simple_imputation_categorical(df, categorical_vars):
    df = df.copy()

    for var in categorical_vars:
        print(f"Processing variable: {var}")
        # Missing mask
        missing_mask = df[var].isnull()
        missing_count = missing_mask.sum()

        if missing_count == 0:
            print(f"No missing values in {var}. Skipping.")
            continue
        else:
            print(f"Missing values in {var}: {missing_count}")

        # Use mode (most frequent value) for categorical variables
        mode_value = df[var].mode()[0]  # Take first mode if there are multiple

        # Impute missing values
        df.loc[missing_mask, var] = mode_value

        print(f"Imputed missing values for {var} using mode value: {mode_value}")

    return df


def logistic_regression_imputation(df, top_correlations, binary_vars):

    models = {}
    df = df.copy()

    for var in binary_vars:
        if var not in top_correlations:
            continue

        print(f"Processing variable: {var}")

        # Missing mask
        missing_mask = df[var] == -1

        # Check if there are missing values
        missing_count = missing_mask.sum()

        if missing_count == 0:
            print(f"No missing values in {var}. Skipping.")
            continue
        else:
            print(f"Missing values in {var}: {missing_count}")

        predictors = top_correlations[var]
        X = df[predictors]
        y = df[var]

        # Valid rows (non-missing in target and all predictors)
        valid_mask = ~missing_mask & X.notnull().all(axis=1)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Impute missing values in predictors
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X_valid)

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Train the model
        model = LogisticRegression(random_state=11)
        try:
            model.fit(X_scaled, y_valid)
            print(f"Model trained successfully for {var}.")
        except Exception as e:
            print(f"Error fitting model for {var}: {e}")
            continue

        # Impute missing values
        if missing_count > 0:
            X_missing = df.loc[missing_mask, predictors]
            X_missing_imputed = imputer.transform(X_missing)
            X_missing_scaled = scaler.transform(X_missing_imputed)
            y_pred = model.predict(X_missing_scaled)
            df.loc[missing_mask, var] = y_pred
            print(f"Imputed {len(y_pred)} values for {var}.")

        # Store the model and preprocessing objects
        models[var] = {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "predictors": predictors,
        }

    return df, models
