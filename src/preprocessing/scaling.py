"""
Standardisation and normalisation of the dataset.

"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


DEFAULT_BLOCKS: Dict[str, List[str]] = { # TODO : modifier les colonnes en fonction de ce qu'il reste
    "chemistry": [
        "C","Si","Mn","S","P","Ni","Cr","Mo","V","Cu","Co","W",
        "O","Ti","N","Al","B","Nb","Sn","As","Sb"
    ],
    "process": [
        "Current","Voltage","AC_DC","ElectrodePolarity","HeatInput",
        "InterpassTemp","WeldType","PWHT_Temp","PWHT_Time"
    ],
    "micro": [
        "FerritePrimary","FerriteSecondPhase","AcicularFerrite",
        "Martensite","FerriteWithCarbides"
    ],
    "mech": [
        "YieldStrength","UTS","Elongation","ReductionArea",
        "CharpyTemp","CharpyImpact","Hardness","FATT50"
    ],
}

SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}

class FeatureScaler:
    """
    FeatureScaler(df)
    
    Handles scaling for the different blocks in a chainable way.
    Fit and training on a subset to avoid any data leak.

    Attributes: 
        df: pd.DataFrame   (muté in-place par transform_all)
        blocks: Dict[str, List[str]]
        policies: Dict[str, str]   (block -> "standard" | "minmax" | "robust" | "none")
        fitted_: Dict[str, object] (block -> scaler fitted)

    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.blocks: Dict[str, List[str]] = {}
        self.policies: Dict[str, str] = {}
        self.fitted_: Dict[str, Optional[TransformerMixin]] = {}
        self._original_df: Optional[pd.DataFrame] = None

    def set_blocks(self, blocks: Optional[Dict[str, List[str]]] = None) -> "FeatureScaler":

        """
        Defines the blocks, filters DEFAULT_BLOCKS on the existing columns by default.
        
        """

        if blocks is None:
            self.blocks = {b: [c for c in cols if c in self.df.columns]
                           for b, cols in DEFAULT_BLOCKS.items()}
        else:
            self.blocks = {b: [c for c in cols if c in self.df.columns]
                           for b, cols in blocks.items()}
            
        return self

    def configure(self, policies: Dict[str, str]) -> "FeatureScaler":

        """
        Associates a scaler to each block, for example chemistry = standard, process = robust, etc.

        """
        self.policies = policies
        return self

    def _fit_block(self, block: str, train_idx: Optional[pd.Index]) -> None:
        """
        Trains the scaler for a given block, step by step. The different steps are: 

        1. Selecting the numerical columns;
        2. Getting the scaling policy for that column;
        3. Instantiating the corresonding scaler class;
        4. Fitting the scaler only to the training subset;
        5. Store the fitted scaler so that it can be used later on all data during the transformation.
        
        """
        cols = [c for c in self.blocks.get(block, []) if pd.api.types.is_numeric_dtype(self.df[c])]
        policy = self.policies.get(block, "none")
        ScalerCls = SCALER_REGISTRY.get(policy, None)

        if not cols:
            self.fitted_[block] = None
            return
        
        if ScalerCls is None:
            self.fitted_[block] = None
            return

        X = self.df.loc[train_idx, cols] if train_idx is not None else self.df[cols]
        self.assert_no_missing(cols=cols)
        scaler = ScalerCls()
        scaler.fit(X.values)
        self.fitted_[block] = scaler

    def _transform_block(self, block: str) -> None:
        """
        Apply the previously fitted scaler to all numerical columns of a given block
        
        """
        cols = [c for c in self.blocks.get(block, []) if pd.api.types.is_numeric_dtype(self.df[c])]
        scaler = self.fitted_.get(block, None)

        if not cols or scaler is None:
            return
        
        self.assert_no_missing(cols=cols)
        self.df[cols] = scaler.transform(self.df[cols].values)

    def fit(self, train_idx: Optional[pd.Index] = None) -> "FeatureScaler":
        """
        Fitting of the scalers by block, only on the training set
        
        """
        for block in self.blocks:
            self._fit_block(block, train_idx=train_idx)

        return self

    def transform_all(self) -> "FeatureScaler":
        """
        Transformation of all blocks
        
        """
        for block in self.blocks:
            self._transform_block(block)

        return self

    def fit_transform(self, train_idx: Optional[pd.Index] = None) -> "FeatureScaler":
        """
        Convenience: fit() then transform_all()
        
        """
        return self.fit(train_idx=train_idx).transform_all()


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    DataFrameScaler(columns, scaler="standard")

    Applies a sklearn scaler to a subset of numerical colums, and returns a dataframe with the original columns transformed where needed.

    Parameters:
    columns : Sequence[str]
    scaler : {"standard","minmax","robust","none"}

    """

    def __init__(self, columns: Sequence[str], scaler: str = "standard"):
        self.columns = list(columns)
        self.scaler = scaler
        self._scaler_: Optional[TransformerMixin] = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fitting the scaler on the specified columns of the dataframe
        
        """

        if self.scaler == "none":
            self._scaler_ = None
            return self
        
        ScalerCls = SCALER_REGISTRY.get(self.scaler, None)

        if ScalerCls is None:
            raise ValueError(f"Unknown scaler '{self.scaler}'.")
        
        self._scaler_ = ScalerCls().fit(X[self.columns].astype(float).values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted scaler to the specified dataframe columns
        
        """

        Xout = X.copy()

        if self._scaler_ is None:
            return Xout
        
        Xout[self.columns] = self._scaler_.transform(Xout[self.columns].astype(float).values)

        return Xout
