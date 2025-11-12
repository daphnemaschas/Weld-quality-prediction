"""
PCA Preprocessing Module

This module provides functions to perform PCA on groups of columns independently.
Each group is processed separately, keeping 90% of variance explained.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict


def apply_pca_by_groups(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column_groups: List[List[str]],
    group_names: List[str] = None,
    target_col: str = "YieldStrength",
    variance_retained: float = 0.90,
    center_only: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Apply PCA independently to each group of columns.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset
    test_df : pd.DataFrame
        Test dataset
    column_groups : List[List[str]]
        List of groups, where each group is a list of column names
    group_names : List[str], optional
        Names for each group (e.g., ['Chemical', 'Mechanical', ...])
        If None, will use 'Group_1', 'Group_2', etc.
    target_col : str
        Name of the target column (will be preserved)
    variance_retained : float
        Proportion of variance to retain (default: 0.90 = 90%)
    center_only : bool
        If True, only center the data (mean=0), don't scale (std=1)
        If False, use StandardScaler (mean=0, std=1)
    
    Returns:
    --------
    train_transformed : pd.DataFrame
        Transformed training dataset
    test_transformed : pd.DataFrame
        Transformed test dataset
    pca_info : Dict
        Dictionary containing PCA information for each group
    """
    
    # Separate target and categorical columns (already encoded)
    # Identify categorical columns (one-hot encoded columns)
    # They have patterns like: AC_DC_AC, AC_DC_DC, ElectrodePolarity_+, WeldType_MMA, etc.
    categorical_cols = []
    for col in train_df.columns:
        if col == target_col:
            continue
        # Check if it's a one-hot encoded categorical column
        if any(col.startswith(prefix) for prefix in ['AC_DC_', 'ElectrodePolarity_', 'WeldType_']):
            categorical_cols.append(col)
    
    # Get all numerical columns that will be processed by PCA
    all_group_cols = []
    for group in column_groups:
        all_group_cols.extend(group)
    
    # Columns to preserve: target + categorical (these won't be touched by PCA)
    preserve_cols = [target_col] + categorical_cols
    preserve_cols = [col for col in preserve_cols if col in train_df.columns]
    
    # Note: Any numerical columns not in column_groups will be excluded from the final dataset
    # This is intentional - we only want the PCA-transformed groups + categorical + target
    
    # Initialize transformed datasets
    train_transformed = train_df[preserve_cols].copy()
    test_transformed = test_df[preserve_cols].copy()
    
    # Dictionary to store PCA information
    pca_info = {
        'groups': {},
        'total_original_cols': 0,
        'total_reduced_cols': 0,
        'total_cols_removed': 0
    }
    
    print("="*80)
    print("PCA BY GROUPS - PROCESSING")
    print("="*80)
    print(f"Target column: {target_col}")
    print(f"Variance retained: {variance_retained*100:.1f}%")
    print(f"Scaling mode: {'Center only (mean=0)' if center_only else 'Standardize (mean=0, std=1)'}")
    print(f"Categorical columns preserved: {len(categorical_cols)}")
    print("="*80)
    
    # Process each group independently
    for group_idx, group_cols in enumerate(column_groups):
        # Filter to only existing columns
        existing_cols = [col for col in group_cols if col in train_df.columns]
        
        if len(existing_cols) == 0:
            group_name = group_names[group_idx] if group_names and group_idx < len(group_names) else f"Group_{group_idx + 1}"
            print(f"\n⚠️  {group_name}: No columns found in dataset")
            continue
        
        # Use provided group name or default
        if group_names and group_idx < len(group_names):
            group_name = group_names[group_idx]
        else:
            group_name = f"Group_{group_idx + 1}"
        print(f"\n{group_name}: {existing_cols[0] if len(existing_cols) == 1 else f'{len(existing_cols)} columns'}")
        print(f"  Columns: {existing_cols}")
        
        # Extract group data
        train_group = train_df[existing_cols].copy()
        test_group = test_df[existing_cols].copy()
        
        # Handle missing values (shouldn't happen in normalized data, but just in case)
        train_group = train_group.fillna(train_group.mean())
        test_group = test_group.fillna(train_group.mean())  # Use train mean for test
        
        original_n_cols = len(existing_cols)
        pca_info['total_original_cols'] += original_n_cols
        
        # Center (and optionally scale) the data
        if center_only:
            # Center only: subtract mean, don't divide by std
            scaler = StandardScaler(with_mean=True, with_std=False)
        else:
            # Standardize: subtract mean and divide by std
            scaler = StandardScaler(with_mean=True, with_std=True)
        
        # Fit scaler on train, transform both train and test
        train_group_scaled = pd.DataFrame(
            scaler.fit_transform(train_group),
            columns=existing_cols,
            index=train_group.index
        )
        test_group_scaled = pd.DataFrame(
            scaler.transform(test_group),
            columns=existing_cols,
            index=test_group.index
        )
        
        # Apply PCA
        pca = PCA(n_components=variance_retained)
        train_group_pca = pca.fit_transform(train_group_scaled)
        test_group_pca = pca.transform(test_group_scaled)
        
        # Get number of components
        n_components = pca.n_components_
        variance_explained = pca.explained_variance_ratio_.sum()
        
        # Create column names for PCA components
        pca_col_names = [f"{group_name}_PC{i+1}" for i in range(n_components)]
        
        # Convert to DataFrame
        train_group_pca_df = pd.DataFrame(
            train_group_pca,
            columns=pca_col_names,
            index=train_group.index
        )
        test_group_pca_df = pd.DataFrame(
            test_group_pca,
            columns=pca_col_names,
            index=test_group.index
        )
        
        # Add PCA components to transformed datasets
        train_transformed = pd.concat([train_transformed, train_group_pca_df], axis=1)
        test_transformed = pd.concat([test_transformed, test_group_pca_df], axis=1)
        
        # Store information
        cols_removed = original_n_cols - n_components
        pca_info['groups'][group_name] = {
            'original_columns': existing_cols,
            'original_n_cols': original_n_cols,
            'n_components': n_components,
            'variance_explained': variance_explained,
            'cols_removed': cols_removed,
            'pca_column_names': pca_col_names
        }
        pca_info['total_reduced_cols'] += n_components
        pca_info['total_cols_removed'] += cols_removed
        
        # Print summary
        print(f"  Original columns: {original_n_cols}")
        print(f"  PCA components: {n_components}")
        print(f"  Columns removed: {cols_removed}")
        print(f"  Variance explained: {variance_explained*100:.2f}%")
    
    print("\n" + "="*80)
    print("PCA SUMMARY")
    print("="*80)
    print(f"Total original columns: {pca_info['total_original_cols']}")
    print(f"Total reduced columns: {pca_info['total_reduced_cols']}")
    print(f"Total columns removed: {pca_info['total_cols_removed']}")
    print(f"Reduction: {pca_info['total_cols_removed']}/{pca_info['total_original_cols']} = "
          f"{pca_info['total_cols_removed']/pca_info['total_original_cols']*100:.1f}%")
    print(f"Final train shape: {train_transformed.shape}")
    print(f"Final test shape: {test_transformed.shape}")
    print("="*80)
    
    return train_transformed, test_transformed, pca_info

