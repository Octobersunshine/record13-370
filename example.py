import numpy as np
from boxcox_service import BoxCoxTransform, BoxCoxVisualizer


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    np.random.seed(42)

    print_section("示例 1: 对数正态分布数据（右偏）")
    lognorm_data = np.random.lognormal(mean=0, sigma=1.0, size=1000)
    print(f"样本数量: {len(lognorm_data)}")
    print(f"原始数据范围: [{lognorm_data.min():.4f}, {lognorm_data.max():.4f}]")

    transformer = BoxCoxTransform()
    result = transformer.compare_transform(lognorm_data)
    transformed_data = transformer.fit_transform(lognorm_data)

    print(f"\n--- 变换前统计 ---")
    print(f"  是否正态: {result['original']['is_normal']}")
    print(f"  Shapiro p值: {result['original']['p_value']:.6f}")
    print(f"  偏度: {result['original']['skewness']:.4f}")
    print(f"  峰度: {result['original']['kurtosis']:.4f}")

    print(f"\n--- 变换后统计 ---")
    print(f"  最优 λ: {result['lambda_value']:.4f}")
    print(f"  是否正态: {result['transformed']['is_normal']}")
    print(f"  Shapiro p值: {result['transformed']['p_value']:.6f}")
    print(f"  偏度: {result['transformed']['skewness']:.4f}")
    print(f"  峰度: {result['transformed']['kurtosis']:.4f}")

    print(f"\n--- 改善指标 ---")
    print(f"  p值提升: {result['p_value_improvement']:.6f}")
    print(f"  偏度降低: {result['skewness_reduction']:.4f}")
    print(f"  峰度降低: {result['kurtosis_reduction']:.4f}")

    print_section("示例 2: 逆变换验证")
    reconstructed = transformer.inverse_transform(transformed_data)
    mae = np.mean(np.abs(lognorm_data - reconstructed))
    print(f"逆变换平均绝对误差: {mae:.10f}")
    print(f"数据完全还原: {np.allclose(lognorm_data, reconstructed, atol=1e-6)}")

    print_section("示例 3: 包含非正数数据（自动偏移）")
    shifted_data = lognorm_data - 3.0
    print(f"数据最小值: {shifted_data.min():.4f}")

    transformer2 = BoxCoxTransform()
    result2 = transformer2.compare_transform(shifted_data)
    print(f"自动偏移量: {result2['shift']:.4f}")
    print(f"最优 λ: {result2['lambda_value']:.4f}")
    print(f"变换后是否正态: {result2['transformed']['is_normal']}")
    print(f"Shapiro p值: {result2['transformed']['p_value']:.6f}")

    reconstructed2 = transformer2.inverse_transform(transformer2.transform(shifted_data))
    print(f"逆变换还原精度: {np.allclose(shifted_data, reconstructed2, atol=1e-6)}")

    print_section("示例 4: 手动指定 Lambda 值")
    transformer3 = BoxCoxTransform(lambda_value=0.1)
    transformed_manual = transformer3.fit_transform(lognorm_data)
    stats_manual = transformer3.normality_test(transformed_manual)
    print(f"指定 λ=0.1 时的 Shapiro p值: {stats_manual['p_value']:.6f}")
    print(f"指定 λ=0.1 时的偏度: {stats_manual['skewness']:.4f}")

    print_section("示例 5: Weibull 分布数据")
    weibull_data = np.random.weibull(a=1.5, size=1000)
    transformer4 = BoxCoxTransform()
    result4 = transformer4.compare_transform(weibull_data)
    print(f"原始 p值: {result4['original']['p_value']:.6f}")
    print(f"最优 λ: {result4['lambda_value']:.4f}")
    print(f"变换后 p值: {result4['transformed']['p_value']:.6f}")
    print(f"变换后是否正态: {result4['transformed']['is_normal']}")

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("提示: 如需查看可视化效果，请安装 matplotlib 并")
    print("调用 BoxCoxVisualizer.plot_comparison()")
    print("=" * 60)


if __name__ == "__main__":
    main()
