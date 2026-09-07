"""
Statistical Analysis Module

Implements all statistical tests used in the paper with detailed documentation
as requested by Reviewer 3.

Author: Dongxing Yu
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional, List
import warnings


class StatisticalAnalysis:
    """
    Statistical analysis tools for the DCMT study.
    
    This class implements all statistical methods mentioned in Table 4
    of the manuscript, with clear documentation for reproducibility.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize with random seed for reproducibility.
        
        Args:
            random_seed: Seed for random operations (default: 42)
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
    
    def cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = False
    ) -> float:
        """
        Calculate Cohen's d effect size.
        
        Cohen's d measures the standardized difference between two means.
        
        Formula:
            d = (M1 - M2) / SD_pooled
            
        Where SD_pooled = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))
        
        Interpretation:
            - 0.2 = small effect
            - 0.5 = medium effect
            - 0.8 = large effect
        
        Args:
            group1: First group data
            group2: Second group data
            paired: Whether samples are paired
            
        Returns:
            Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        m1, m2 = np.mean(group1), np.mean(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        if paired:
            diff = group1 - group2
            sd = np.std(diff, ddof=1)
            d = np.mean(diff) / sd
        else:
            pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            d = (m1 - m2) / pooled_sd
        
        return d
    
    def fleiss_kappa(
        self,
        ratings: np.ndarray
    ) -> float:
        """
        Calculate Fleiss' Kappa for inter-rater agreement.
        Formula: κ = (P_o - P_e) / (1 - P_e)
        """
        n_items, n_categories = ratings.shape
        n_raters = ratings.sum(axis=1)[0]
        
        p_j = ratings.sum(axis=0) / (n_items * n_raters)
        p_i = (ratings**2).sum(axis=1) - n_raters
        p_i = p_i / (n_raters * (n_raters - 1))
        p_bar = p_i.mean()
        p_e = (p_j**2).sum()
        kappa = (p_bar - p_e) / (1 - p_e)
        return kappa
    
    def paired_ttest(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        bonferroni_n: int = 1
    ) -> Tuple[float, float, float]:
        """Perform paired t-test with optional Bonferroni correction."""
        t_stat, p_value = stats.ttest_rel(group1, group2)
        p_adjusted = min(p_value * bonferroni_n, 1.0)
        return t_stat, p_value, p_adjusted
    
    def pearson_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bootstrap_n: int = 10000
    ) -> Tuple[float, float, Tuple[float, float]]:
        """Calculate Pearson correlation with bootstrap confidence interval."""
        r, p = stats.pearsonr(x, y)
        bootstrap_rs = []
        n = len(x)
        for _ in range(bootstrap_n):
            indices = np.random.choice(n, n, replace=True)
            r_boot, _ = stats.pearsonr(x[indices], y[indices])
            bootstrap_rs.append(r_boot)
        ci_lower = np.percentile(bootstrap_rs, 2.5)
        ci_upper = np.percentile(bootstrap_rs, 97.5)
        return r, p, (ci_lower, ci_upper)
    
    def kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """Calculate Kullback-Leibler divergence between distributions."""
        p = np.array(p) / np.sum(p)
        q = np.array(q) / np.sum(q)
        p = p + epsilon
        q = q + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))
    
    def mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: int = 20
    ) -> float:
        """Estimate mutual information between two variables (bits)."""
        c_xy = np.histogram2d(x, y, bins=bins)[0]
        p_xy = c_xy / np.sum(c_xy)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi
    
    def f_test_variance(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Tuple[float, float]:
        """Perform F-test for equality of variances."""
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        f_stat = var1 / var2
        df1 = len(group1) - 1
        df2 = len(group2) - 1
        p_value = 2 * min(
            stats.f.cdf(f_stat, df1, df2),
            1 - stats.f.cdf(f_stat, df1, df2)
        )
        return f_stat, p_value


def print_statistical_summary():
    """Print summary of statistical methods used in the paper."""
    summary = """
    ============================================================
    STATISTICAL METHODS SUMMARY (Table 4 in manuscript)
    ============================================================
    
    1. Cohen's d - Effect size (0.2=small, 0.5=medium, 0.8=large)
    2. Fleiss' κ - Inter-rater agreement (>0.75=excellent)
    3. Pearson's r - Correlation (±0.1=weak, ±0.3=moderate, ±0.5=strong)
    4. Bonferroni correction - Multiple comparisons
    5. KL divergence - Distribution similarity
    6. Mutual Information - Shared information (bits)
    ============================================================
    """
    print(summary)


if __name__ == "__main__":
    print_statistical_summary()
