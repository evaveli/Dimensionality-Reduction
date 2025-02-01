import pandas as pd
import numpy as np
from preprocessing.data_imputation import (
    simple_imputation,
    simple_imputation_categorical,
    logistic_regression_imputation,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from preprocessing.data_analysis import get_top_correlations
from utils import check_missing_values


class Preprocessing:
    def __init__(self):
        self.binary_vars = []
        self.categorical_vars = []
        self.continuous_vars = []
        self.missing_values = False

    def generous_preprocessing(self, df):
        # Change encoding, remove uninformative columns
        df = self.change_coding(df)

        y_true = df.iloc[:, -1].to_numpy()
        if isinstance(y_true[0], str) and set(y_true) == {"LIVE", "DIE"}:
            y_true = np.where(y_true == "LIVE", 0, 1)
        else:
            # For other datasets, ensure integer type
            y_true = y_true.astype(int)

        df = self.remove_class_labels(df)

        # Identify variable types
        self.binary_vars, self.categorical_vars, self.continuous_vars = (
            self.identify_variable_types(df)
        )

        # Handle encoding
        df = self.handle_encoding(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Apply scalers
        df1 = self.scale_dataset(df)
        df2 = self.robustscale_dataset(df)
        df3 = self.standard_scaler_dataset(df)

        # Create a numoy array for better computing for all scaled datasets
        X = df1.to_numpy()
        X_robust = df2.to_numpy()
        X_standard = df3.to_numpy()

        return X, X_robust, X_standard, y_true

    def handle_missing_values(self, df):
        if check_missing_values(df):
            df = self.impute_missing_values(df)

        return df

    def impute_missing_values(self, df):

        df_imputed = df.copy()
        print(self.binary_vars)
        if self.binary_vars:
            matrix = df_imputed.corr()
            top_correlations = get_top_correlations(
                matrix,
                exclude_vars=["Class"],
                n=5,
            )
            df_imputed, models = logistic_regression_imputation(
                df_imputed, top_correlations, self.binary_vars
            )
            # You might want to store the models somewhere if needed
            self.binary_imputation_models = models

        if self.continuous_vars:
            df_imputed = simple_imputation(df_imputed, self.continuous_vars)
        if self.categorical_vars:
            df_imputed = simple_imputation_categorical(
                df_imputed, self.categorical_vars
            )
        return df_imputed

    def change_coding(self, df):
        for col in df.columns:
            if df[col].dtype == object:
                if df[col].apply(lambda x: isinstance(x, bytes)).any():
                    df[col] = df[col].str.decode("utf-8", errors="ignore")
                # If it's not bytes, it might be a former categorical column
                else:
                    df[col] = df[col].astype(str)
            elif "category" in str(df[col].dtype):
                df[col] = df[col].astype(str)

        return df

    def handle_encoding(self, df):

        if len(self.binary_vars) > 0:
            df = self.recode_dataset(df)

        # Handle one-hot encoding if needed
        if len(self.categorical_vars) > 0:
            print("Applying One-Hot Encoding to categorical variables.")
            # Fit and transform data
            ohe = OneHotEncoder()
            ohe_features = ohe.fit_transform(df[self.categorical_vars])
            ohe_df = pd.DataFrame(
                ohe_features.toarray(),
                columns=ohe.get_feature_names_out(self.categorical_vars),
                index=df.index,
            )
            df = df.drop(columns=self.categorical_vars)
            df = pd.concat([df, ohe_df], axis=1)
        else:
            print("No categorical variables to encode.")
        return df

    def remove_class_labels(self, df):
        # Remove the last column being a class label
        df = df.iloc[:, :-1]
        return df

    def identify_variable_types(self, df):
        binary_vars = self.identify_binary_variables(df)
        categorical_vars = self.identify_categorical_variables(df, binary_vars)
        continuous_vars = list(
            set(df.columns) - set(binary_vars) - set(categorical_vars)
        )
        return binary_vars, categorical_vars, continuous_vars

    def identify_binary_variables(self, df):
        binary_vars = []
        for col in df.columns:
            unique_values = (
                df[col].dropna().astype(str).str.replace("^b", "", regex=True).unique()
            )
            # Exclude missing value indicators
            unique_values = [v for v in unique_values if v not in ["?", "nan", "-1"]]
            if len(unique_values) == 2:
                binary_vars.append(col)
        return binary_vars

    def identify_categorical_variables(self, df, binary_vars):
        categorical_vars = []

        for col in df.columns:
            unique_values = df[col].nunique()
            print(
                f"Column: {col}, Type: {df[col].dtype}, Unique Values: {unique_values}, "
                f"Binary: {col in binary_vars}, Class: {col.lower() == 'class'}"
            )

            # Check if it's 'object' dtype or has a low number of unique values (e.g., <10)
            if (
                df[col].dtype == "object" or unique_values < 10
            ) and col not in binary_vars:
                categorical_vars.append(col)

        return categorical_vars

    def frequency(self, df):

        columns_to_process = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if "origin" not in columns_to_process and "origin" in df.columns:
            columns_to_process.append("origin")

        frames = []
        for column in columns_to_process:
            series = df[column].astype(str)
            value_counts = series.value_counts(dropna=False).reset_index()
            value_counts.columns = ["Value", "Frequency"]
            value_counts["Variable"] = column
            frames.append(value_counts[["Variable", "Value", "Frequency"]])
        return pd.concat(frames, ignore_index=True)

    def recode_dataset(self, df):

        freq_df = self.frequency(df)
        columns_to_process = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for column in columns_to_process:
            column_freq = freq_df[freq_df["Variable"] == column].sort_values(
                "Frequency", ascending=False
            )
            recode_map = {}
            code = 0
            for _, row in column_freq.iterrows():
                if row["Value"] == "?":
                    recode_map[row["Value"]] = -1
                else:
                    recode_map[row["Value"]] = code
                    code += 1
            df[column] = df[column].map(recode_map)
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    def scale_dataset(self, df):

        # Create a copy to avoid modifying the original
        scaled_df = df.copy()

        # Get all features to scale (all columns)
        features_to_scale = scaled_df.columns

        # Convert all columns to numeric, replacing non-numeric with NaN
        for column in features_to_scale:
            scaled_df[column] = pd.to_numeric(scaled_df[column], errors="coerce")

        # Initialize and apply the scaler
        scaler = MinMaxScaler()
        scaled_df[features_to_scale] = scaler.fit_transform(
            scaled_df[features_to_scale]
        )

        return scaled_df

    def robustscale_dataset(self, df):

        # Create a copy to avoid modifying the original
        scaled_df = df.copy()

        # Get all features to scale (all columns)
        features_to_scale = scaled_df.columns

        # Convert all columns to numeric, replacing non-numeric with NaN
        for column in features_to_scale:
            scaled_df[column] = pd.to_numeric(scaled_df[column], errors="coerce")

        # Initialize and apply the scaler
        scaler = RobustScaler()
        scaled_df[features_to_scale] = scaler.fit_transform(
            scaled_df[features_to_scale]
        )

        return scaled_df

    def standard_scaler_dataset(self, df):

        # Create a copy to avoid modifying the original
        scaled_df = df.copy()

        # Get all features to scale (all columns)
        features_to_scale = scaled_df.columns

        # Convert all columns to numeric, replacing non-numeric with NaN
        for column in features_to_scale:
            scaled_df[column] = pd.to_numeric(scaled_df[column], errors="coerce")

        # Initialize and apply the scaler
        scaler = StandardScaler()
        scaled_df[features_to_scale] = scaler.fit_transform(
            scaled_df[features_to_scale]
        )

        return scaled_df
