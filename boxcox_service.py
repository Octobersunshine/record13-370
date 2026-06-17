import numpy as np
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from typing import Tuple, Optional, Dict, Any
import warnings


def _yeojohnson_transform(data: np.ndarray, lmbda: float) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    y = np.zeros_like(data)

    pos_mask = data >= 0
    neg_mask = ~pos_mask

    if abs(lmbda) > 1e-10:
        y[pos_mask] = (np.power(data[pos_mask] + 1.0, lmbda) - 1.0) / lmbda
    else:
        y[pos_mask] = np.log(data[pos_mask] + 1.0)

    if abs(lmbda - 2.0) > 1e-10:
        y[neg_mask] = -(np.power(-data[neg_mask] + 1.0, 2.0 - lmbda) - 1.0) / (2.0 - lmbda)
    else:
        y[neg_mask] = -np.log(-data[neg_mask] + 1.0)

    return y


def _inv_yeojohnson(transformed: np.ndarray, lmbda: float) -> np.ndarray:
    transformed = np.asarray(transformed, dtype=float)
    y = np.zeros_like(transformed)

    pos_mask = transformed >= 0
    neg_mask = ~pos_mask

    if abs(lmbda) > 1e-10:
        y[pos_mask] = np.power(lmbda * transformed[pos_mask] + 1.0, 1.0 / lmbda) - 1.0
    else:
        y[pos_mask] = np.exp(transformed[pos_mask]) - 1.0

    if abs(lmbda - 2.0) > 1e-10:
        y[neg_mask] = 1.0 - np.power((2.0 - lmbda) * (-transformed[neg_mask]) + 1.0, 1.0 / (2.0 - lmbda))
    else:
        y[neg_mask] = 1.0 - np.exp(-transformed[neg_mask])

    return y


class BoxCoxTransform:
    def __init__(self, lambda_value: Optional[float] = None):
        self.lambda_value = lambda_value
        self._lambda_user_provided = lambda_value is not None
        self.shift = 0.0
        self._is_fitted = False

    def _validate_positive(self, data: np.ndarray) -> None:
        if np.any(data <= 0):
            raise ValueError(
                "Box-Cox 变换要求所有数据必须为正数。"
                "请使用 add_shift 参数自动添加偏移量，或手动预处理数据。"
            )

    def _auto_shift(self, data: np.ndarray) -> np.ndarray:
        min_val = np.min(data)
        if min_val <= 0:
            self.shift = abs(min_val) + 1e-8
            return data + self.shift
        self.shift = 0.0
        return data

    def fit(
        self,
        data: np.ndarray,
        add_shift: bool = True,
        method: str = "mle"
    ) -> "BoxCoxTransform":
        data = np.asarray(data, dtype=float).flatten()

        if add_shift:
            data = self._auto_shift(data)
        else:
            self.shift = 0.0
            self._validate_positive(data)

        if not self._lambda_user_provided:
            self.lambda_value = stats.boxcox_normmax(data, method=method)

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法拟合模型。")

        data = np.asarray(data, dtype=float).flatten()
        data = data + self.shift
        self._validate_positive(data)

        return boxcox(data, self.lambda_value)

    def fit_transform(
        self,
        data: np.ndarray,
        add_shift: bool = True,
        method: str = "mle"
    ) -> np.ndarray:
        self.fit(data, add_shift=add_shift, method=method)
        return self.transform(data)

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法拟合模型。")

        transformed_data = np.asarray(transformed_data, dtype=float).flatten()
        data = inv_boxcox(transformed_data, self.lambda_value)
        return data - self.shift

    def normality_test(
        self,
        data: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float).flatten()

        stat, p_value = stats.shapiro(data)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        return {
            "shapiro_statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": p_value > alpha,
            "alpha": alpha,
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "n_samples": len(data)
        }

    def compare_transform(
        self,
        data: np.ndarray,
        add_shift: bool = True,
        method: str = "mle",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float).flatten()

        original_stats = self.normality_test(data, alpha=alpha)
        transformed_data = self.fit_transform(data, add_shift=add_shift, method=method)
        transformed_stats = self.normality_test(transformed_data, alpha=alpha)

        return {
            "original": original_stats,
            "transformed": transformed_stats,
            "lambda_value": float(self.lambda_value),
            "shift": float(self.shift),
            "p_value_improvement": float(transformed_stats["p_value"] - original_stats["p_value"]),
            "skewness_reduction": float(abs(original_stats["skewness"]) - abs(transformed_stats["skewness"])),
            "kurtosis_reduction": float(abs(original_stats["kurtosis"]) - abs(transformed_stats["kurtosis"]))
        }


class YeoJohnsonTransform:
    def __init__(self, lambda_value: Optional[float] = None):
        self.lambda_value = lambda_value
        self._lambda_user_provided = lambda_value is not None
        self._is_fitted = False

    def fit(
        self,
        data: np.ndarray,
        brack: Optional[Tuple[float, float]] = None
    ) -> "YeoJohnsonTransform":
        data = np.asarray(data, dtype=float).flatten()

        if not self._lambda_user_provided:
            if brack is not None:
                self.lambda_value = stats.yeojohnson_normmax(data, brack=brack)
            else:
                self.lambda_value = stats.yeojohnson_normmax(data)

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法拟合模型。")

        data = np.asarray(data, dtype=float).flatten()
        return _yeojohnson_transform(data, self.lambda_value)

    def fit_transform(
        self,
        data: np.ndarray,
        brack: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        self.fit(data, brack=brack)
        return self.transform(data)

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法拟合模型。")

        transformed_data = np.asarray(transformed_data, dtype=float).flatten()
        return _inv_yeojohnson(transformed_data, self.lambda_value)

    def normality_test(
        self,
        data: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float).flatten()

        stat, p_value = stats.shapiro(data)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        return {
            "shapiro_statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": p_value > alpha,
            "alpha": alpha,
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "n_samples": len(data)
        }

    def compare_transform(
        self,
        data: np.ndarray,
        brack: Optional[Tuple[float, float]] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        data = np.asarray(data, dtype=float).flatten()

        original_stats = self.normality_test(data, alpha=alpha)
        transformed_data = self.fit_transform(data, brack=brack)
        transformed_stats = self.normality_test(transformed_data, alpha=alpha)

        return {
            "original": original_stats,
            "transformed": transformed_stats,
            "lambda_value": float(self.lambda_value),
            "p_value_improvement": float(transformed_stats["p_value"] - original_stats["p_value"]),
            "skewness_reduction": float(abs(original_stats["skewness"]) - abs(transformed_stats["skewness"])),
            "kurtosis_reduction": float(abs(original_stats["kurtosis"]) - abs(transformed_stats["kurtosis"]))
        }


class BoxCoxVisualizer:
    @staticmethod
    def plot_comparison(
        data: np.ndarray,
        transformed_data: np.ndarray,
        lambda_value: float,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("可视化功能需要安装 matplotlib: pip install matplotlib")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        axes[0, 0].hist(data, bins=30, edgecolor="black", alpha=0.7, color="#3498db")
        axes[0, 0].set_title("原始数据分布")
        axes[0, 0].set_xlabel("数值")
        axes[0, 0].set_ylabel("频数")

        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("原始数据 Q-Q 图")
        axes[0, 1].get_lines()[0].set_color("#3498db")

        axes[1, 0].hist(transformed_data, bins=30, edgecolor="black", alpha=0.7, color="#e74c3c")
        axes[1, 0].set_title(f"Box-Cox 变换后分布 (λ={lambda_value:.4f})")
        axes[1, 0].set_xlabel("数值")
        axes[1, 0].set_ylabel("频数")

        stats.probplot(transformed_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("变换后 Q-Q 图")
        axes[1, 1].get_lines()[0].set_color("#e74c3c")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_lambda_search(
        data: np.ndarray,
        add_shift: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("可视化功能需要安装 matplotlib: pip install matplotlib")

        data = np.asarray(data, dtype=float).flatten()

        if add_shift:
            min_val = np.min(data)
            if min_val <= 0:
                shift = abs(min_val) + 1e-8
                data = data + shift

        lmd_range = np.linspace(-2, 2, 100)
        llf_values = []

        for lmd in lmd_range:
            llf = stats.boxcox_llf(lmd, data)
            llf_values.append(llf)

        best_lmd = stats.boxcox_normmax(data)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lmd_range, llf_values, color="#2c3e50", linewidth=2)
        ax.axvline(best_lmd, color="#e74c3c", linestyle="--", label=f"最佳 λ={best_lmd:.4f}")
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("对数似然值")
        ax.set_title("Box-Cox Lambda 参数搜索")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class YeoJohnsonVisualizer:
    @staticmethod
    def plot_comparison(
        data: np.ndarray,
        transformed_data: np.ndarray,
        lambda_value: float,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("可视化功能需要安装 matplotlib: pip install matplotlib")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        axes[0, 0].hist(data, bins=30, edgecolor="black", alpha=0.7, color="#3498db")
        axes[0, 0].set_title("原始数据分布")
        axes[0, 0].set_xlabel("数值")
        axes[0, 0].set_ylabel("频数")

        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("原始数据 Q-Q 图")
        axes[0, 1].get_lines()[0].set_color("#3498db")

        axes[1, 0].hist(transformed_data, bins=30, edgecolor="black", alpha=0.7, color="#27ae60")
        axes[1, 0].set_title(f"Yeo-Johnson 变换后分布 (λ={lambda_value:.4f})")
        axes[1, 0].set_xlabel("数值")
        axes[1, 0].set_ylabel("频数")

        stats.probplot(transformed_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("变换后 Q-Q 图")
        axes[1, 1].get_lines()[0].set_color("#27ae60")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_lambda_search(
        data: np.ndarray,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("可视化功能需要安装 matplotlib: pip install matplotlib")

        data = np.asarray(data, dtype=float).flatten()

        lmd_range = np.linspace(-2, 2, 100)
        llf_values = []

        for lmd in lmd_range:
            llf = stats.yeojohnson_llf(lmd, data)
            llf_values.append(llf)

        best_lmd = stats.yeojohnson_normmax(data)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lmd_range, llf_values, color="#2c3e50", linewidth=2)
        ax.axvline(best_lmd, color="#27ae60", linestyle="--", label=f"最佳 λ={best_lmd:.4f}")
        ax.set_xlabel("Lambda (λ)")
        ax.set_ylabel("对数似然值")
        ax.set_title("Yeo-Johnson Lambda 参数搜索")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


__all__ = ["BoxCoxTransform", "BoxCoxVisualizer", "YeoJohnsonTransform", "YeoJohnsonVisualizer"]
