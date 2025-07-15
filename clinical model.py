# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from scipy.stats import norm
from imblearn.over_sampling import SMOTEN

# 设置随机种子保证结果可复现
SEED = 42
np.random.seed(SEED)

# 设置字体（如果需要）
plt.rcParams['font.family'] = 'Times New Roman'


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTEN
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

# 随机种子，确保结果可复现
SEED = 42

# -------------------------------
# 1. 读取数据
# -------------------------------
file_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\clinical\clinical-model-svi.xlsx"
data = pd.read_excel(file_path)

# -------------------------------
# 2. 数据预处理
# -------------------------------
target_col = 'label'
X = data.drop(columns=[target_col])
y = data[target_col]

# 全部特征均为分类变量，进行LabelEncoder编码
le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# -------------------------------
# 3. 划分训练集和测试集（分层抽样）
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# -------------------------------
# 4. 使用 SMOTEN 进行过采样（适用于纯分类数据）
# -------------------------------
smoten = SMOTEN(random_state=SEED)
X_train_resampled, y_train_resampled = smoten.fit_resample(X_train, y_train)

print(f"Resampled training set shape: {X_train_resampled.shape}, {y_train_resampled.shape}")

# -------------------------------
# 5. 定义随机森林参数空间与交叉验证方案
# -------------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# -------------------------------
# 6. 网格搜索寻找最佳参数
# -------------------------------
rf = RandomForestClassifier(random_state=SEED, class_weight='balanced')
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc_ovr_weighted',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_resampled, y_train_resampled)
print("Best parameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_

# -------------------------------
# 7. 校准模型概率输出
# -------------------------------
try:
    calibrated_clf = CalibratedClassifierCV(base_estimator=best_rf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_train_resampled, y_train_resampled)
except TypeError:
    calibrated_clf = CalibratedClassifierCV(estimator=best_rf, method='sigmoid', cv=5)
    calibrated_clf.fit(X_train_resampled, y_train_resampled)

# -------------------------------
# 8. 模型预测
# -------------------------------
y_pred = calibrated_clf.predict(X_test)
y_proba = calibrated_clf.predict_proba(X_test)

# -------------------------------
# 9. 计算并输出各类评价指标
# -------------------------------

print("Classification Report:\n", classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

roc_auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
roc_auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
roc_auc_weighted = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

print(f"ROC AUC Score (Macro Average): {roc_auc_macro:.4f}")
print(f"ROC AUC Score (Micro Average): {roc_auc_micro:.4f}")
print(f"ROC AUC Score (Weighted Average): {roc_auc_weighted:.4f}")

metrics = ['macro', 'micro', 'weighted']

for avg in metrics:
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg)
    print(f"\nMetrics ({avg} average):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")



def bootstrap_ci(y_true, y_pred, metric_func, average='macro', n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(SEED)
    n_size = len(y_true)
    scores = []
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else np.array(y_true)

    for _ in range(n_bootstrap):
        indices = rng.choice(np.arange(n_size), size=n_size, replace=True)
        y_true_bs = y_true_arr[indices]
        y_pred_bs = y_pred[indices]

        precision, recall, f1, _ = metric_func(y_true_bs, y_pred_bs, average=average)
        scores.append((precision, recall, f1))

    scores = np.array(scores)
    mean_scores = scores.mean(axis=0)
    lower = np.percentile(scores, (1 - ci) / 2 * 100, axis=0)
    upper = np.percentile(scores, (1 + ci) / 2 * 100, axis=0)

    return mean_scores, lower, upper

mean_scores, lower, upper = bootstrap_ci(y_test, y_pred, precision_recall_fscore_support, average='macro', n_bootstrap=1000, ci=0.95)

print("\nMacro-average metrics with 95% CI (bootstrap):")
print(f"Precision: {mean_scores[0]:.2f} (95% CI: {lower[0]:.2f} - {upper[0]:.2f})")
print(f"Recall:    {mean_scores[1]:.2f} (95% CI: {lower[1]:.2f} - {upper[1]:.2f})")
print(f"F1-score:  {mean_scores[2]:.2f} (95% CI: {lower[2]:.2f} - {upper[2]:.2f})")

# -------------------------------
# 9.2 计算准确率和ROC AUC的95%置信区间（bootstrap）
# -------------------------------
def bootstrap_metric_ci(y_true, y_pred, y_proba, metric_func, n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(SEED)
    n_size = len(y_true)
    scores = []

    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else np.array(y_true)

    for _ in range(n_bootstrap):
        indices = rng.choice(np.arange(n_size), size=n_size, replace=True)
        y_true_bs = y_true_arr[indices]
        y_pred_bs = y_pred[indices]
        y_proba_bs = y_proba[indices] if y_proba is not None else None

        score = metric_func(y_true_bs, y_pred_bs, y_proba_bs)
        scores.append(score)

    scores = np.array(scores)
    mean_score = scores.mean()
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)

    return mean_score, lower, upper

def accuracy_metric(y_true, y_pred, y_proba=None):
    return accuracy_score(y_true, y_pred)

def roc_auc_metric(y_true, y_pred, y_proba):
    classes_in_sample = np.unique(y_true)
    class_indices = [cls for cls in classes_in_sample]
    y_proba_sub = y_proba[:, class_indices]

    if len(classes_in_sample) > 2:
        return roc_auc_score(
            y_true, y_proba_sub,
            multi_class='ovr',
            average='macro',
            labels=classes_in_sample
        )
    elif len(classes_in_sample) == 2:
        pos_label = classes_in_sample[1]
        pos_class_idx = class_indices.index(pos_label)
        y_proba_pos = y_proba_sub[:, pos_class_idx]
        return roc_auc_score(
            y_true, y_proba_pos,
            average='macro',
            labels=classes_in_sample
        )
    else:
        # 只有一个类别，无法计算AUC，返回np.nan（注意：不是None）
        return np.nan

acc_mean, acc_lower, acc_upper = bootstrap_metric_ci(y_test, y_pred, None, accuracy_metric, n_bootstrap=1000, ci=0.95)
print(f"\nAccuracy (95% CI): {acc_mean:.2f} ({acc_lower:.2f} - {acc_upper:.2f})")

auc_mean, auc_lower, auc_upper = bootstrap_metric_ci(y_test, y_pred, y_proba, roc_auc_metric, n_bootstrap=1000, ci=0.95)
print(f"ROC AUC Macro Average (95% CI): {auc_mean:.2f} ({auc_lower:.2f} - {auc_upper:.2f})")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from matplotlib.patches import FancyBboxPatch

# 假设您已有：
# y_test, y_pred, y_proba, class_names（例如：np.unique(y_test)）
class_names = np.unique(y_test)

# 1. 计算每个类别的指标
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=class_names
)
accuracy_per_class = recall_per_class  # 这里用召回率近似每类准确率

# 计算每个类别的AUC（one-vs-rest）
y_test_bin = label_binarize(y_test, classes=class_names)
auc_per_class = []
for i in range(len(class_names)):
    auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
    auc_per_class.append(auc)
auc_per_class = np.array(auc_per_class)

# 2. 组合指标
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
metrics_per_class = np.vstack([
    accuracy_per_class,
    precision_per_class,
    recall_per_class,
    f1_per_class,
    auc_per_class
]).T  # 形状为 (类别数, 指标数)

# 打印每个类别的指标
print(f"{'类别':<10} " + " ".join([f"{name:>10}" for name in metrics_names]))
for idx, cls in enumerate(class_names):
    vals = metrics_per_class[idx]
    print(f"{cls:<10} " + " ".join([f"{v:10.3f}" if not np.isnan(v) else '    nan   ' for v in vals]))
    
# 3. 自定义图例名称和颜色（长度应等于类别数）
# 如果您知道类别对应的中文名或其他描述，可自定义
custom_legend_names = ['very high-risk','high-risk','low-risk']  # 请自行根据类别数调整

# 如果类别数超过您自定义颜色数，自动从colormap获取
base_colors = ['#e74c3c', '#3498db', '#f39c12']  # 您的自定义颜色
if len(class_names) > len(base_colors):
    cmap = plt.cm.get_cmap('tab10', len(class_names))
    colors = [cmap(i) for i in range(len(class_names))]
else:
    colors = base_colors[:len(class_names)]

def plot_radar(metrics, class_labels, metric_labels, legend_names=None, colors=None,
               legend_loc='upper center', title_bg_width=1.0,
               title_bg_linewidth=1.5, title_bg_color='gray',
               title_y=1.0,fontsize=14):
    """
    参数说明：
    - title_bg_width: 圆角矩形宽度，取值0~1，1表示画布全宽
    - title_bg_linewidth: 圆角矩形边框线宽
    - title_bg_color: 圆角矩形边框颜色
    - title_y: 标题纵坐标（figure坐标系），控制标题和圆角矩形整体高度
    """

    num_vars = len(metric_labels)
    # 五边形角度（顶点），从90度开始旋转，使顶部为第一顶点
    sides = 5
    pentagon_angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.pi / 2

    # 这里需要用正五边形顶点数，若metric_labels多于5个，取前5个绘制标签
    # 如果metric_labels不止5个，建议只显示5个（或自行调整）
    if num_vars != sides:
        print(f"警告：metric_labels长度({num_vars})不等于5，绘制时只使用前5个标签")
    display_labels = metric_labels[:sides]

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

    if colors is None:
        colors = plt.cm.get_cmap('tab10', len(class_labels)).colors
    if legend_names is None:
        legend_names = class_labels

    # 绘制雷达图数据
    for idx, row in enumerate(metrics):
        vals = row[:sides].tolist()  # 只取5个指标
        vals += vals[:1]  # 闭合
        angles_plot = np.append(pentagon_angles, pentagon_angles[0])
        ax.plot(angles_plot, vals, color=colors[idx], linewidth=2, label=str(legend_names[idx]))
        ax.scatter(angles_plot, vals, color=colors[idx], s=50, zorder=10, linewidth=0.5)

    # 设置极轴参数
    ax.set_theta_offset(0)
    ax.set_theta_direction(1)

    # 关闭默认极坐标网格和边框
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_frame_on(False)
    ax.patch.set_visible(False)

    # 设置极轴范围
    ax.set_rlim(0, 1)

    # 不显示默认极坐标刻度标签
    ax.set_yticklabels([])

     # 关闭极坐标角度刻度标签（theta刻度）
    ax.set_xticks([])  # 这行代码关闭外围角度刻度显示

    # 自定义绘制多层正五边形网格
    grid_radii = np.linspace(0.2, 1.0, 5)
    for r in grid_radii:
        xs = r * np.cos(pentagon_angles)
        ys = r * np.sin(pentagon_angles)
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        ax.plot(np.arctan2(ys, xs), np.hypot(xs, ys), color='gray', linestyle='dotted', linewidth=1, zorder=0)

    # 绘制正五边形边框（最外围）
    xs = np.cos(pentagon_angles)
    ys = np.sin(pentagon_angles)
    xs = np.append(xs, xs[0])
    ys = np.append(ys, ys[0])
    ax.plot(np.arctan2(ys, xs), np.hypot(xs, ys), color='gray', linestyle='-', linewidth=1.5, zorder=5)

     # 添加径向数值刻度文本（在顶部角度方向）
    r_ticks = grid_radii
    r_tick_labels = [f"{r:.1f}" for r in r_ticks]
    angle_for_labels = np.pi / 2  # 顶部顶点方向角度

    for r, label in zip(r_ticks, r_tick_labels):
        ax.text(angle_for_labels, r, label,
                fontsize=14,
                color='gray',
                horizontalalignment='center',
                verticalalignment='bottom',
                rotation=0,
                rotation_mode='anchor')

    # **添加径向网格线（从中心到各顶点）**
    for angle in pentagon_angles:
        ax.plot([angle, angle], [0, 1], color='gray', linestyle='dotted', linewidth=1, zorder=0)

    # 角度标签放置，标签位置稍微超出边界
    label_r = 1.1  # 标签半径，稍大于边框半径1
    for angle, label in zip(pentagon_angles, display_labels):
        x = label_r * np.cos(angle)
        y = label_r * np.sin(angle)
        ax.text(angle, label_r, label, fontsize=18,
                horizontalalignment='center',
                verticalalignment='center')

    # 添加图例
    legend = ax.legend(
        loc=legend_loc,
        bbox_to_anchor=(0.5, 1.13),
        ncol=len(class_labels),
        fontsize=18,
        frameon=False
    )
    for text, color in zip(legend.get_texts(), colors):
        text.set_color(color)

    # 添加标题及圆角矩形
    title_text = "Clinical Model"
    title = fig.suptitle(title_text, fontsize=18, y=title_y, color='#16a085',fontweight='bold')

    fig.canvas.draw()
    bbox = title.get_window_extent(fig.canvas.get_renderer())
    inv = fig.transFigure.inverted()
    bbox_fig = bbox.transformed(inv)

    pad_y = 0.01
    rect_y = bbox_fig.y0 - pad_y

    rect_height = bbox_fig.height + 2 * pad_y
    rect_width = max(0, min(title_bg_width, 1.0))
    rect_x = 0.5 - rect_width / 2

    rect = FancyBboxPatch(
        (rect_x, rect_y),
        rect_width,
        rect_height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=title_bg_linewidth,
        edgecolor=title_bg_color,
        facecolor='none',
        transform=fig.transFigure,
        zorder=2
    )
    fig.patches.append(rect)
    title.set_zorder(3)
    plt.tight_layout()
    output_path = r"C:\Users\ACER\Desktop\test\stt\picture\Clinical\Clinical Model.jpg"
    plt.savefig(output_path, dpi=600, format='jpg',bbox_inches='tight')
    plt.show()


custom_legend_dict = {
    0: 'very high-risk',
    1: 'high-risk',
    2: 'low-risk',
    # 如果有更多类别，继续添加
}

legend_names = [custom_legend_dict.get(cls, str(cls)) for cls in class_names]

plot_radar(
    metrics=metrics_per_class,
    class_labels=class_names,
    metric_labels=metrics_names,
    legend_names=legend_names,
    colors=base_colors,
    legend_loc='upper center',
    title_bg_width=0.65,      # 圆角矩形宽度，单位figure坐标系
    title_bg_linewidth=2,
    title_bg_color='#16a085',
    title_y=1.02,
    # 标题及矩形整体位置，调低使其向下移动
)

from sklearn.utils import resample

def bootstrap_metric_CI(y_true, y_pred, y_proba, class_names, n_bootstrap=1000, alpha=0.05, random_state=None):
    """
    计算每个类别指标的95%置信区间（Accuracy/Precision/Recall/F1/AUC）基于Bootstrap。

    参数：
    - y_true: 真实标签，shape (n_samples,)
    - y_pred: 预测标签，shape (n_samples,)
    - y_proba: 预测概率，shape (n_samples, n_classes)
    - class_names: 类别列表
    - n_bootstrap: bootstrap次数
    - alpha: 显著水平（一般0.05，95%CI）
    - random_state: 随机种子
    
    返回：
    - metrics_CI: dict，键为指标名称，值为shape (n_classes, 2)的置信区间（下界、上界）
    """

    np.random.seed(random_state)
    n_samples = len(y_true)

    # 初始化存储bootstrap指标的数组
    boot_metrics = {
        'Accuracy': np.zeros((n_bootstrap, len(class_names))),
        'Precision': np.zeros((n_bootstrap, len(class_names))),
        'Recall': np.zeros((n_bootstrap, len(class_names))),
        'F1-score': np.zeros((n_bootstrap, len(class_names))),
        'AUC': np.zeros((n_bootstrap, len(class_names)))
    }

    # 真实标签二值化
    y_true_bin = label_binarize(y_true, classes=class_names)

    for i in range(n_bootstrap):
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        y_true_bs = y_true.iloc[indices] if isinstance(y_true, pd.Series) else y_true[indices]
        y_pred_bs = y_pred.iloc[indices] if isinstance(y_pred, pd.Series) else y_pred[indices]
        y_proba_bs = y_proba[indices]  # 通常是numpy数组
    # 然后计算指标...


        # 计算指标
        p, r, f1, _ = precision_recall_fscore_support(y_true_bs, y_pred_bs, average=None, labels=class_names)
        recall_bs = r  # 用召回率近似Accuracy（如你之前）
        # AUC逐类别计算
        y_true_bin_bs = label_binarize(y_true_bs, classes=class_names)
        auc_bs = []
        for c in range(len(class_names)):
            try:
                auc_val = roc_auc_score(y_true_bin_bs[:, c], y_proba_bs[:, c])
            except Exception:
                auc_val = np.nan
            auc_bs.append(auc_val)
        auc_bs = np.array(auc_bs)

        # 存储
        boot_metrics['Accuracy'][i, :] = recall_bs
        boot_metrics['Precision'][i, :] = p
        boot_metrics['Recall'][i, :] = r
        boot_metrics['F1-score'][i, :] = f1
        boot_metrics['AUC'][i, :] = auc_bs

    # 计算置信区间
    metrics_CI = {}
    for metric_name, values in boot_metrics.items():
        lower = np.nanpercentile(values, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(values, 100 * (1 - alpha / 2), axis=0)
        metrics_CI[metric_name] = np.vstack((lower, upper)).T  # shape (n_classes, 2)

    return metrics_CI

metrics_CI = bootstrap_metric_CI(y_test, y_pred, y_proba, class_names, n_bootstrap=1000, alpha=0.05, random_state=42)

# 打印每个类别指标及置信区间示例
print(f"{'类别':<10} {'指标':<10} {'估计值':>10} {'95%CI':>20}")
for i, cls in enumerate(class_names):
    print(f"类别 {cls}:")
    for metric_idx, metric_name in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']):
        est = metrics_per_class[i, metric_idx]
        ci_lower, ci_upper = metrics_CI[metric_name][i]
        print(f"  {metric_name:<10} {est:10.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")



import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np

# 假设你的y_test类别编码如下
# 0, 1, 2

# 定义类别标签映射
category_labels = ['very high-risk', 'high-risk', 'low-risk']

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# 绘制热力图
plt.figure(figsize=(12,8))
ax = sns.heatmap(cm_norm, annot=False, fmt='.2%', cmap='Blues', xticklabels=category_labels, yticklabels=category_labels)

# 添加详细标注
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percent = cm_norm[i, j]
        ax.text(j + 0.5, i + 0.5, f'{count}\n({percent:.2%})',
                ha='center', va='center', color='black', fontsize=20)

plt.title('Confusion Matrix',fontsize=18)
plt.xlabel('Predicted',fontsize=18)
plt.ylabel('True',fontsize=18)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
# 保存图片
save_path = r"C:\Users\ACER\Desktop\test\stt\picture\Clinical\Confusion Matrix.jpg"
plt.savefig(save_path, dpi=900)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample

# 计算单个类别ROC AUC及95% CI的Bootstrap函数
def bootstrap_auc_ci(y_true_bin, y_score, n_bootstraps=1000, alpha=0.05, random_seed=42):
    rng = np.random.RandomState(random_seed)
    aucs = []
    n_samples = len(y_true_bin)
    
    for i in range(n_bootstraps):
        indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        if len(np.unique(y_true_bin[indices])) < 2:
            # 该bootstrap样本中只包含单一类别，跳过
            continue
        score = roc_auc_score(y_true_bin[indices], y_score[indices])
        aucs.append(score)
    
    aucs = np.array(aucs)
    mean_auc = np.mean(aucs)
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return mean_auc, lower, upper


# 自定义类别名称映射（原类别0,1,2对应新的显示名）
label_map = {0: 'very high-risk', 1: 'high-risk', 2: 'low-risk'}
colors = ['#e74c3c', '#3498db', '#f39c12']

fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
y_test_binarized = pd.get_dummies(y_test).values  # 转成numpy方便索引
y_test_index = y_test.reset_index(drop=True)      # 重置索引方便

plt.figure(figsize=(12,8))
legend_handles = []

for i, class_label in enumerate(label_map.keys()):
    y_true_bin = y_test_binarized[:, i]
    y_score = y_proba[:, i]
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    fpr_dict[class_label], tpr_dict[class_label] = fpr, tpr
    
    # 计算AUC及95% CI
    mean_auc, lower_ci, upper_ci = bootstrap_auc_ci(y_true_bin, y_score, n_bootstraps=1000)
    
    # 画线
    color = colors[i]
    display_name = label_map[class_label]
    plt.plot(fpr, tpr, lw=2, color=color,
             label=f'{display_name} (AUC = {mean_auc:.2f} [{lower_ci:.2f}, {upper_ci:.2f}])')
    
    # 保存legend handle
    handle = plt.Line2D([], [], color=color, lw=2,
                        label=f'{display_name} (AUC = {mean_auc:.2f} [{lower_ci:.2f}, {upper_ci:.2f}])')
    legend_handles.append(handle)

# 计算宏平均ROC曲线（用已有方法）
all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in fpr_dict]))
mean_tpr = np.zeros_like(all_fpr)
for c in fpr_dict:
    mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
mean_tpr /= len(fpr_dict)

# 宏平均AUC及95%CI，顺序与上文一致
# 注意宏平均AUC需要用整个y_test_binarized和y_proba计算Bootstrap
def bootstrap_macro_auc_ci(y_true_bin_matrix, y_score_matrix, n_bootstraps=1000, alpha=0.05):
    rng = np.random.RandomState(42)
    aucs = []
    n_samples = y_true_bin_matrix.shape[0]
    
    for i in range(n_bootstraps):
        indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        if np.any(np.sum(y_true_bin_matrix[indices], axis=0) == 0):
            # 某个类别在采样中缺失，跳过
            continue
        auc = roc_auc_score(y_true_bin_matrix[indices], y_score_matrix[indices], average='macro', multi_class='ovr')
        aucs.append(auc)
    aucs = np.array(aucs)
    mean_auc = aucs.mean()
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return mean_auc, lower, upper

roc_auc_macro, lower_macro, upper_macro = bootstrap_macro_auc_ci(y_test_binarized, y_proba, n_bootstraps=1000)
plt.plot(all_fpr, mean_tpr, color='#8e44ad', lw=2,
         label=f'Macro-average ROC (AUC = {roc_auc_macro:.2f} [{lower_macro:.2f}, {upper_macro:.2f}])')
macro_handle = plt.Line2D([], [], color='#8e44ad', lw=2,
                          label=f'Macro-average ROC (AUC = {roc_auc_macro:.2f} [{lower_macro:.2f}, {upper_macro:.2f}])')
legend_handles.append(macro_handle)

plt.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d', alpha=0.6)
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Multi-class ROC Curves', fontsize=18)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.legend(handles=legend_handles, loc='lower right', fontsize=16)

save_path = r"C:\Users\ACER\Desktop\test\stt\picture\Clinical\Multi-class ROC Curves.jpg"
plt.savefig(save_path, dpi=600, format='jpg')
plt.show()


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义使用的特征
model_features = ['SVI', 'Tumour Burden']

# 提取测试集对应特征
X_test_aligned = X_test[model_features]

# 计算SHAP值
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_aligned)  # 长度为类别数的列表

# 获取类别标签
category_list = best_rf.classes_

# 计算每个类别的平均绝对SHAP值
category_shap_importance = {}
for i, category in enumerate(category_list):
    shap_vals = shap_values[i]  # shape: (特征数, 样本数)
    mean_abs_shap = np.abs(shap_vals).mean(axis=1)
    category_shap_importance[category] = mean_abs_shap

# 转换为DataFrame，行索引为特征，列为类别
shap_importance_df = pd.DataFrame(category_shap_importance, index=model_features)

# 反转顺序（可选）
shap_importance_df = shap_importance_df.iloc[::-1]

# 按每列排序
sorted_shap_df = shap_importance_df.copy()
for col in sorted_shap_df.columns:
    sorted_shap_df[col] = sorted_shap_df[col].sort_values(ascending=False).values

# 全局排序：每个特征的平均重要性
total_importance = sorted_shap_df.mean(axis=1)
sorted_features = total_importance.sort_values(ascending=False).index
sorted_shap_df = sorted_shap_df.loc[sorted_features]

# 颜色定义
colors = ['#3c9bc8', '#b8ddb1', '#fee59e'][:len(category_list)]
category_names = ['very high-risk', 'high-risk', 'low-risk']  # 你可以自定义

# 绘制堆叠条形图
fig, ax = plt.subplots(figsize=(12, 8))
sorted_shap_df.plot(
    kind='barh',
    stacked=True,
    color=colors,
    ax=ax,
    legend=False  # 先关闭默认图例，后面自定义
)

# 获取句柄和标签
handles, labels = ax.get_legend_handles_labels()
# 使用自定义类别名设置图例
plt.legend(handles, category_names, title='ADT Resistant Risk', loc='lower right',fontsize=16, title_fontsize=14)

# 添加每个条形堆叠的数值标签
for i, feature in enumerate(sorted_shap_df.index):
    y = i  # y轴位置
    cumulative = 0
    for j, category in enumerate(sorted_shap_df.columns):
        value = sorted_shap_df.loc[feature, category]
        y_offset = cumulative + value / 2
        cumulative += value
        ax.text(
            y_offset,
            y,
            f'{value:.2f}',
            va='center',
            ha='center',
            fontsize=18,
            color='black'
        )

plt.title('Feature Importance by Class',fontsize=20)
plt.xlabel('Mean Absolute SHAP Value',fontsize=18)
plt.ylabel('Features',fontsize=18)
ax.invert_yaxis()
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=18)
plt.tight_layout()
plt.savefig(r"C:\Users\ACER\Desktop\test\stt\picture\Clinical\Feature Importance by Class.jpg", dpi=600)
plt.show()

# 绘制全局特征重要性
avg_shap_values = np.mean([np.abs(sv).mean(axis=1) for sv in shap_values], axis=0)
sorted_indices = np.argsort(-avg_shap_values)
sorted_features_global = [model_features[i] for i in sorted_indices]
sorted_avg_shap = avg_shap_values[sorted_indices]

plt.figure(figsize=(12, 8))
plt.barh(sorted_features_global, sorted_avg_shap)
plt.xlabel('Mean Absolute SHAP Value',fontsize=18)
plt.ylabel('Features',fontsize=18)
plt.title('Global SHAP Feature Importance',fontsize=20)
plt.gca().invert_yaxis()
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=18)
plt.tight_layout()
plt.savefig(r"C:\Users\ACER\Desktop\test\stt\picture\Clinical\Global SHAP Feature Importance.jpg", dpi=600)
plt.show()




