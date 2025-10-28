"""Data cleaning utilities for the Weld-quality-prediction project.

This module provides a `MissingValueHandler` class and a helper function
`initial_cleanup` to perform light type coercion and missing-value
preparation prior to EDA or pipeline integration.

The class offers methods to:
- inspect missingness (`get_missing_summary`, `show_missing_summary`)
- visualize missingness patterns (`plot_missingness`)
- coerce column types (`convert_types`)
- apply a variety of imputation strategies (`impute_with_strategies`,
  `simple_impute_for_process_params`, `knn_impute`, `iterative_impute`)
- domain-specific cleaning (`clean_chemical_columns`, `clean_and_impute_mechanical`)
- create missingness indicators (`mark_mnar`)
- drop rows/columns by missingness thresholds (`drop_by_threshold`)
- evaluate simple imputation strategies (`evaluate_imputation`)
- persist the cleaned dataframe (`save_dataframe`)
"""

from typing import Dict, Optional, Sequence, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, accuracy_score  # type: ignore
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


class MissingValueHandler:
    """
    Inspect, visualize and impute missing values in a pandas DataFrame.

    The handler stores a reference to a DataFrame in `self.df` and most
    operations modify it *in place*. Methods typically return `self`
    to allow method chaining in interactive sessions.

    Example:
        df = pd.read_csv("data.csv")
        df = initial_cleanup(df)
        mvh = MissingValueHandler(df)
        mvh.show_missing_summary()
        mvh.impute_with_strategies({'colA': 'median', 'colB': 'mode'})

    Attributes:
        df (pd.DataFrame): The working DataFrame being inspected and modified.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the handler with a DataFrame.

        Args:
            df: A pandas DataFrame to operate on. The DataFrame is stored
                as `self.df` and mutated by the handler's methods.
        """
        self.df = df


    def get_missing_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of missingness statistics per column.

        The returned DataFrame contains:
            - 'total_missing': absolute number of missing values
            - 'pct_missing': percentage of missing values in [0, 100]
            - 'dtype': column dtype (as string)
            - 'n_unique': number of unique (non-NA) values

        Returns:
            A pandas DataFrame sorted by descending pct_missing.
        """
        total_missing = self.df.isna().sum()
        pct_missing = 100 * self.df.isna().mean()
        dtype_series = self.df.dtypes.astype(str)
        n_unique = self.df.nunique(dropna=True)

        summary_df = pd.DataFrame({
            'total_missing': total_missing,
            'pct_missing': pct_missing,
            'dtype': dtype_series,
            'n_unique': n_unique
        })

        return summary_df.sort_values('pct_missing', ascending=False)

    def show_missing_summary(self, top: int = 50) -> pd.DataFrame:
        """Print and return a compact missingness summary.

        Args:
            top: Number of top columns (by missingness) to print and return.

        Returns:
            The top-N rows of the missingness summary DataFrame.
        """
        summary_df = self.get_missing_summary()
        top_df = summary_df.head(top).copy()
        print(top_df.to_string())
        return top_df

    def plot_missingness(self, method: str = 'heatmap', sample_rows: Optional[int] = 2000) -> None:
        """Visualize missing-value patterns.

        Supported visualization methods:
            - 'bar': horizontal barplot of percent missing (top columns)
            - 'matrix': matrix visualization using `missingno` (optional)
            - 'heatmap': correlation heatmap of missingness indicators (default)

        Args:
            method: Visualization type. One of {'bar', 'matrix', 'heatmap'}.
            sample_rows: Maximum number of rows to sample when using the 'matrix'
                visualization to avoid rendering extremely large matrices.
        """
        sns.set()

        if method == 'bar':
            summary_df = self.get_missing_summary().reset_index()
            limited = summary_df.head(40)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=limited, x='pct_missing', y='index')
            plt.xlabel('Percent missing')
            plt.ylabel('Column')
            plt.title('Top missing columns')
            plt.tight_layout()
            plt.show()
            return

        if method == 'matrix':
            try: # TODO
                import missingno as msno  # optional dependency
                sample_df = self.df.sample(min(len(self.df), sample_rows), random_state=0)
                msno.matrix(sample_df)
                plt.show()
                return
            except Exception:
                # If missingno is not available, fall back to heatmap.
                print('missingno not available, falling back to heatmap')

        # Default: heatmap of missingness correlation
        miss_corr = self.df.isna().astype(int).corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(miss_corr, cmap='viridis', vmin=-1, vmax=1)
        plt.title('Correlation of missingness between columns')
        plt.tight_layout()
        plt.show()

    def convert_types(self,
                      schema: Optional[Dict[str, str]] = None,
                      infer_numeric_from_objects: bool = True) -> pd.DataFrame:
        """Perform initial type coercion and heuristic conversions.

        Behavior:
            - If `schema` is provided, apply exact conversions for listed columns.
              Supported kinds: 'datetime', 'numeric', 'category', 'bool'.
            - If `schema` is None, heuristics are applied:
                * attempt to parse date-like column names into datetime
                * coerce object columns that look numeric into numeric dtype
                * convert low-cardinality object columns to 'category'

        Args:
            schema: Optional mapping column -> kind (see above).
            infer_numeric_from_objects: Whether to try converting object columns
                that look numeric into numeric dtype.

        Returns:
            The cleaned DataFrame (same object stored in `self.df`).
        """
        dataframe = self.df

        if schema:
            # Apply explicit schema conversions
            for col, kind in schema.items():
                if col not in dataframe.columns:
                    continue
                if kind == 'datetime':
                    dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')
                elif kind == 'numeric':
                    # Remove common thousands separators and coerce
                    dataframe[col] = pd.to_numeric(
                        dataframe[col].astype(str).str.replace(',', '').str.strip(),
                        errors='coerce'
                    )
                elif kind == 'category':
                    dataframe[col] = dataframe[col].astype('category')
                elif kind == 'bool':
                    dataframe[col] = dataframe[col].astype('bool')
        else:
            # Heuristic 1: parse columns with date/time hints
            date_hints = [c for c in dataframe.columns if 'date' in c.lower() or 'time' in c.lower()]
            for col in date_hints:
                try:
                    dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')
                except Exception:
                    # If parsing fails, keep original column
                    pass

            # Heuristic 2: coerce object columns that are mostly numeric
            if infer_numeric_from_objects:
                for col in dataframe.select_dtypes(include=['object']).columns:
                    sample = dataframe[col].dropna().astype(str)
                    if len(sample) == 0:
                        continue
                    sample_small = sample.sample(min(len(sample), 500), random_state=0)
                    numeric_count = sample_small.apply(
                        lambda x: x.replace(',', '').replace(' ', '').lstrip('+-').isdigit()
                    ).sum()
                    if (numeric_count / len(sample_small)) >= 0.9:
                        dataframe[col] = pd.to_numeric(
                            dataframe[col].astype(str).str.replace(',', '').str.strip(),
                            errors='coerce'
                        )

            # Heuristic 3: low-cardinality object -> category
            for col in dataframe.select_dtypes(include=['object']).columns:
                if dataframe[col].nunique(dropna=True) < 50:
                    dataframe[col] = dataframe[col].astype('category')

        self.df = dataframe
        return dataframe

    def impute_with_strategies(self, strategy_map: Dict[str, str]) -> 'MissingValueHandler':
        """Apply simple imputation strategies defined in strategy_map.

        Supported strategies:
            - 'median': numeric median
            - 'mean': numeric mean
            - 'mode': most frequent value
            - 'ffill': forward fill
            - 'bfill': backward fill
            - 'drop': drop the column

        Args:
            strategy_map: Mapping from column name to strategy string.

        Returns:
            self (for chaining).
        """
        dataframe = self.df
        for col, strategy in strategy_map.items():
            if col not in dataframe.columns:
                continue
            if strategy == 'median' and pd.api.types.is_numeric_dtype(dataframe[col]):
                dataframe[col].fillna(dataframe[col].median(), inplace=True)
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(dataframe[col]):
                dataframe[col].fillna(dataframe[col].mean(), inplace=True)
            elif strategy == 'mode':
                mode_series = dataframe[col].mode()
                if not mode_series.empty:
                    dataframe[col].fillna(mode_series.iloc[0], inplace=True)
            elif strategy == 'ffill':
                dataframe[col].fillna(method='ffill', inplace=True)
            elif strategy == 'bfill':
                dataframe[col].fillna(method='bfill', inplace=True)
            elif strategy == 'drop':
                dataframe.drop(columns=[col], inplace=True)

        self.df = dataframe
        return self

    def simple_impute_for_process_params(self,
                                         process_param_columns: Optional[Sequence[str]] = None,
                                         drop_low_missing: Optional[Sequence[str]] = None,
                                         low_missing_threshold: float = 0.01) -> 'MissingValueHandler':
        """
        Apply heuristic imputation for process parameters.

        Heuristics:
            - numeric process parameters -> median imputation (SimpleImputer)
            - categorical process parameters -> most frequent (SimpleImputer)
            - optionally drop columns with extremely low counts (user choice)

        Args:
            process_param_columns: Optional list of columns to consider; defaults
                to a pre-chosen list of typical process parameters.
            drop_low_missing: Optional list of columns to consider for dropping
                based on low missing fraction.
            low_missing_threshold: Columns with missing fraction below this value
                (for those listed in drop_low_missing) will be dropped.

        Returns:
            self (for chaining).
        """
        dataframe = self.df

        # Default process parameter columns typically encountered in the dataset
        if process_param_columns is None:
            process_param_columns = [
                'Current', 'Voltage', 'AC_DC', 'ElectrodePolarity',
                'HeatInput', 'InterpassTemp', 'WeldType', 'PWHT_Temp', 'PWHT_Time'
            ]

        if drop_low_missing is None:
            drop_low_missing = ['PWHT_Temp', 'PWHT_Time']

        # Numeric median imputation for present numeric columns
        numeric_cols = [
            c for c in process_param_columns
            if c in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[c])
        ]
        if numeric_cols:
            imp_num = SimpleImputer(strategy='median')
            dataframe[numeric_cols] = imp_num.fit_transform(dataframe[numeric_cols])

        # Categorical most-frequent imputation for present categorical columns
        cat_cols = [
            c for c in process_param_columns
            if c in dataframe.columns and not pd.api.types.is_numeric_dtype(dataframe[c])
        ]
        if cat_cols:
            imp_cat = SimpleImputer(strategy='most_frequent')
            dataframe[cat_cols] = imp_cat.fit_transform(dataframe[cat_cols])

        # Optionally drop columns with very low counts (user choice)
        for col in drop_low_missing:
            if col in dataframe.columns:
                pct_missing = dataframe[col].isna().mean()
                if pct_missing < low_missing_threshold:
                    dataframe.drop(columns=[col], inplace=True)

        self.df = dataframe
        return self

    def knn_impute(self, numeric_cols: Optional[Sequence[str]] = None, n_neighbors: int = 5) -> 'MissingValueHandler':
        """Impute numeric columns using KNNImputer.

        Args:
            numeric_cols: List of numeric columns to impute. If None, all numeric
                columns are used.
            n_neighbors: Number of neighbors for the KNNImputer.

        Returns:
            self (for chaining).
        """
        dataframe = self.df
        if numeric_cols is None:
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_arr = imputer.fit_transform(dataframe[numeric_cols])
        dataframe[numeric_cols] = imputed_arr
        self.df = dataframe
        return self

    def iterative_impute_columns(self, columns: Sequence[str], max_iter: int = 10) -> 'MissingValueHandler':
        """Run IterativeImputer on the given numeric columns.

        Args:
            columns: Sequence of column names to impute (non-numeric columns are ignored).
            max_iter: Maximum number of iterations for the IterativeImputer.

        Returns:
            self (for chaining).
        """
        dataframe = self.df
        cols = [c for c in columns if c in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[c])]
        if not cols:
            return self
        imputer = IterativeImputer(max_iter=max_iter, random_state=0)
        imputed_arr = imputer.fit_transform(dataframe[cols])
        dataframe[cols] = imputed_arr
        self.df = dataframe
        return self

    def iterative_impute(self, numeric_cols: Optional[Sequence[str]] = None, max_iter: int = 10) -> 'MissingValueHandler':
        """Impute numeric columns using IterativeImputer (MICE).

        Args:
            numeric_cols: If None, all numeric columns are imputed.
            max_iter: Maximum number of iterations.

        Returns:
            self (for chaining).
        """
        dataframe = self.df
        if numeric_cols is None:
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()

        imputer = IterativeImputer(max_iter=max_iter, random_state=0)
        imputed_arr = imputer.fit_transform(dataframe[numeric_cols])
        dataframe[numeric_cols] = imputed_arr
        self.df = dataframe
        return self

    def clean_chemical_columns(self, chem_cols: Optional[Sequence[str]] = None) -> 'MissingValueHandler':
        """Clean chemistry-related columns (decimal separators, ND tokens, units).

        Behavior:
            - replace comma decimal separator with dot
            - remove tokens like 'ND', 'nd', 'Na', 'NA', and angle brackets
            - coerce to numeric, turning non-parsable entries into NaN
            - print a terse log showing additional NaNs introduced by coercion

        Args:
            chem_cols: Sequence of chemistry column names to process. If None,
                a default list common in the dataset is used.

        Returns:
            self (for chaining).
        """
        dataframe = self.df
        if chem_cols is None:
            chem_cols = [
                'C', 'Si', 'Mn', 'S', 'P', 'Ni', 'Cr', 'Mo', 'V', 'Cu', 'Co',
                'W', 'O', 'Ti', 'N', 'Al', 'B', 'Nb', 'Sn', 'As', 'Sb'
            ]

        present_cols = [col for col in chem_cols if col in dataframe.columns]
        for col in present_cols:
            # Convert to string, replace comma decimals, remove ND-like tokens and angle brackets
            tmp = dataframe[col].astype(str)
            tmp = tmp.str.replace(',', '.', regex=False)
            tmp = tmp.str.replace(r'[<>]|ND|nd|Na|NA', '', regex=True)
            coerced = pd.to_numeric(tmp.replace('', np.nan), errors='coerce')
            before_na = dataframe[col].isna().sum()
            after_na = coerced.isna().sum()
            dataframe[col] = coerced
            # Informational log: how many NaNs were added by coercion
            print(f'clean_chemical_columns: {col} coerced, added {after_na - before_na} NaNs')

        self.df = dataframe
        return self

    def clean_and_impute_mechanical(self,
                                    mech_cols: Optional[Sequence[str]] = None,
                                    create_flag: bool = True,
                                    max_iter: int = 10) -> 'MissingValueHandler':
        """
        Clean mechanical properties columns and optionally impute them.

        Steps:
            - attempt to extract numeric hardness from strings with units
            - optionally create a binary `MechanicalTestDone` flag indicating
              whether any mechanical property is present for a row
            - run iterative imputation on the provided mechanical columns

        Args:
            mech_cols: List of mechanical columns to consider. If None, a sensible
                default list is used.
            create_flag: If True, add a 'MechanicalTestDone' indicator column.
            max_iter: Maximum iterations for iterative imputation.

        Returns:
            self (for chaining).
        """
        dataframe = self.df
        if mech_cols is None:
            mech_cols = [
                'YieldStrength', 'UTS', 'Elongation', 'ReductionArea',
                'CharpyTemp', 'CharpyImpact', 'Hardness', 'FATT50'
            ]

        present = [c for c in mech_cols if c in dataframe.columns]

        # Extract numeric portion of Hardness if present (strip units like 'HV' etc.)
        if 'Hardness' in present:
            try:
                dataframe['Hardness'] = (
                    dataframe['Hardness'].astype(str)
                    .str.extract(r'([0-9]*\.?[0-9]+)')[0]
                    .astype(float)
                )
            except Exception:
                # If extraction fails, keep the column as-is
                pass

        # Create MechanicalTestDone binary flag if requested
        if create_flag:
            if present:
                dataframe['MechanicalTestDone'] = (~dataframe[present].isna().all(axis=1)).astype(int)

        # Run iterative imputation on the mechanical columns (if available)
        try:
            self.df = dataframe
            self.iterative_impute_columns(present, max_iter=max_iter)
        except RuntimeError:
            print('Iterative imputer not available (scikit-learn required). Mechanical imputation skipped.')

        self.df = dataframe
        return self

    def mark_mnar(self, columns: Sequence[str]) -> 'MissingValueHandler':
        """Create binary missingness indicator columns for given columns.

        For each column name `col` in `columns`, a new column named
        `{col}_missing` is created, equal to 1 where the original value is
        missing and 0 otherwise.

        Args:
            columns: Sequence of column names to mark.

        Returns:
            self (for chaining).
        """
        dataframe = self.df
        for col in columns:
            if col in dataframe.columns:
                dataframe[f'{col}_missing'] = dataframe[col].isna().astype(int)
        self.df = dataframe
        return self

    def drop_by_threshold(self, col_thresh: float = 0.5, row_thresh: float = 0.5) -> 'MissingValueHandler':
        """Drop columns and rows exceeding missingness thresholds.

        Args:
            col_thresh: Drop columns with fraction missing >= col_thresh.
            row_thresh: Drop rows with fraction missing >= row_thresh.

        Returns:
            self (for chaining).
        """
        dataframe = self.df
        col_missing_frac = dataframe.isna().mean()
        cols_to_drop = col_missing_frac[col_missing_frac >= col_thresh].index.tolist()
        if cols_to_drop:
            dataframe.drop(columns=cols_to_drop, inplace=True)

        row_missing_frac = dataframe.isna().mean(axis=1)
        rows_to_drop = row_missing_frac[row_missing_frac >= row_thresh].index
        if len(rows_to_drop) > 0:
            dataframe.drop(index=rows_to_drop, inplace=True)

        self.df = dataframe
        return self

    def evaluate_imputation(self,
                            holdout_cols: Sequence[str],
                            mask_frac: float = 0.1,
                            strategy: str = 'median') -> pd.DataFrame:
        """Evaluate a simple imputation strategy by masking observed values.

        Procedure:
            - For each column in `holdout_cols`, randomly mask `mask_frac` of
              observed values, perform the chosen imputation on the masked
              dataset, and compute an error metric:
                * numeric columns -> mean squared error (MSE)
                * non-numeric columns -> accuracy (string comparison)

        Args:
            holdout_cols: Columns to evaluate.
            mask_frac: Fraction of observed values to mask for evaluation.
            strategy: Simple strategy to use ('median' or 'mean' for numeric).

        Returns:
            A DataFrame with per-column evaluation (col, metric, score).
        """
        results = []
        dataframe = self.df

        for col in holdout_cols:
            if col not in dataframe.columns:
                continue
            series = dataframe[col]
            if series.isna().all():
                continue

            observed = series.dropna().sample(frac=mask_frac, random_state=0)
            idx = observed.index
            true_vals = series.loc[idx]
            temp_df = dataframe.copy()
            temp_df.loc[idx, col] = np.nan

            # Apply simple strategy
            if pd.api.types.is_numeric_dtype(series):
                if strategy == 'median':
                    temp_df[col].fillna(temp_df[col].median(), inplace=True)
                elif strategy == 'mean':
                    temp_df[col].fillna(temp_df[col].mean(), inplace=True)
                score = mean_squared_error(true_vals, temp_df.loc[idx, col])
                results.append({'col': col, 'metric': 'mse', 'score': score})
            else:
                mode_series = temp_df[col].mode()
                fill_value = mode_series.iloc[0] if not mode_series.empty else np.nan
                temp_df[col].fillna(fill_value, inplace=True)
                acc = (temp_df.loc[idx, col].astype(str) == true_vals.astype(str)).mean()
                results.append({'col': col, 'metric': 'accuracy', 'score': acc})

        return pd.DataFrame(results)

    def save_dataframe(self, path: str, format: str = 'parquet') -> None:
        """Save the current DataFrame to disk.

        Args:
            path: File path to write to.
            format: 'parquet' (default) or 'csv'.

        Raises:
            Any IO-related exceptions are propagated from pandas.
        """
        if format == 'parquet':
            self.df.to_parquet(path)
        else:
            self.df.to_csv(path, index=False)


def initial_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Perform a light initial cleanup useful before studying missing values.

    The function:
        - strips whitespace in string/object columns
        - converts obvious numeric-looking object columns to numeric
        - parses date/time-like columns based on column names
        - converts low-cardinality object columns to category
        - attempts a safe downcast of numeric columns to float where possible

    Args:
        df: Input DataFrame (will be copied).

    Returns:
        A cleaned copy of the DataFrame.
    """
    dataframe = df.copy()

    # Strip whitespace from object columns and normalize string 'nan' to np.nan
    for col in dataframe.select_dtypes(include=['object']).columns:
        dataframe[col] = dataframe[col].astype(str).str.strip().replace({'nan': np.nan})

    # Parse columns whose names include 'date' or 'time'
    for col in dataframe.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')
            except Exception:
                # Parsing failure: leave column as-is
                pass

    # Coerce object columns that look numeric (sample-based heuristic)
    for col in dataframe.select_dtypes(include=['object']).columns:
        sample_series = dataframe[col].dropna().astype(str)
        if len(sample_series) == 0:
            continue
        sample_small = sample_series.sample(min(len(sample_series), 500), random_state=0)
        numeric_count = sample_small.apply(
            lambda x: x.replace(',', '').replace(' ', '').lstrip('+-').isdigit()
        ).sum()
        if (numeric_count / len(sample_small)) >= 0.9:
            dataframe[col] = pd.to_numeric(
                dataframe[col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            )

    # Convert low-cardinality object columns to category
    for col in dataframe.select_dtypes(include=['object']).columns:
        if dataframe[col].nunique(dropna=True) < 50:
            dataframe[col] = dataframe[col].astype('category')

    # Downcast numeric columns to float where safe to reduce memory
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        try:
            dataframe[col] = pd.to_numeric(dataframe[col], downcast='float')
        except Exception:
            pass

    return dataframe