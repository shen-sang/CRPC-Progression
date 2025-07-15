import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.over_sampling import ADASYN, SMOTENC
from catboost import CatBoostClassifier

plt.rcParams['font.family'] = 'Times New Roman'

random_seed = 42
np.random.seed(random_seed)

# 读取数据
dataFile = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\combine.xlsx"
data = pd.read_excel(dataFile)

target = 'label'
feature_cols = [col for col in data.columns if col != target]

categorical_features = ['Tumour Burden','SVI']
numeric_features = [col for col in feature_cols if col not in categorical_features]

X_categorical = data[categorical_features]
X_numeric = data[numeric_features]
y = data[target]

X_all = pd.concat([X_numeric, X_categorical], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.3, random_state=random_seed, stratify=y
)

num_features_len = X_numeric.shape[1]

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[:, :num_features_len])
X_test_num_scaled = scaler.transform(X_test[:, :num_features_len])

X_train_processed = np.hstack([X_train_num_scaled, X_train[:, num_features_len:]])
X_test_processed = np.hstack([X_test_num_scaled, X_test[:, num_features_len:]])

categorical_feature_indices = list(range(num_features_len, X_train_processed.shape[1]))

adasyn = ADASYN(random_state=random_seed)
smotenc = SMOTENC(categorical_features=categorical_feature_indices, random_state=random_seed)

try:
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_processed, y_train)
    print("ADASYN重采样成功。")
except Exception as e:
    print(f"ADASYN重采样失败，原因：{e}。尝试使用SMOTENC。")
    X_train_resampled, y_train_resampled = smotenc.fit_resample(X_train_processed, y_train)
    print("SMOTENC重采样成功。")

models = {
    "Logistic Regression": LogisticRegression(multi_class='ovr', max_iter=5000, random_state=random_seed),
    "SVM": SVC(probability=True, kernel='rbf', decision_function_shape='ovr', random_state=random_seed),
    "Random Forest": RandomForestClassifier(random_state=random_seed),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(random_state=random_seed),
    "Gradient Boosting": GradientBoostingClassifier(random_state=random_seed),
    "CatBoost": CatBoostClassifier(silent=True, task_type='CPU', random_seed=random_seed)
}

param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    },
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    "Naive Bayes": {},
    "AdaBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "CatBoost": {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [3, 5]
    }
}

best_models = {}
results = []

print("开始模型调参与训练...")
for model_name, model in models.items():
    print(f"训练 {model_name} ...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5,
                               scoring='roc_auc_ovr', n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_models[model_name] = grid_search.best_estimator_
    results.append({
        "Model": model_name,
        "Best Parameters": grid_search.best_params_,
        "Best CV Macro AUC": grid_search.best_score_
    })

results_df = pd.DataFrame(results)
print("模型调优结果：")
print(results_df)
results_df.to_excel(r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\model_tuning_results-SVI.xlsx", index=False)

# -------------------------
# Bootstrap置信区间计算函数
def bootstrap_ci(y_true, y_pred_label, y_pred_proba, metric_func, 
                 n_bootstraps=1000, alpha=0.05, multi_class=False, **metric_kwargs):
    """
    Bootstrap法估计指标的置信区间
    :param y_true: array-like 真正标签
    :param y_pred_label: array-like 预测标签
    :param y_pred_proba: array-like 预测概率，shape (n_samples, n_classes)
    :param metric_func: 计算指标的函数，接受(y_true, y_pred)或(y_true, y_pred_prob)
    :param n_bootstraps: 采样次数
    :param alpha: 置信水平
    :param multi_class: 是否多分类roc_auc_score
    :param metric_kwargs: 传递给指标函数的额外参数
    :return: (均值, 下限, 上限)
    """
    rng = np.random.RandomState(random_seed)
    scores = []
    y_true = np.array(y_true)
    y_pred_label = np.array(y_pred_label)
    y_pred_proba = np.array(y_pred_proba)
    n_samples = len(y_true)
    unique_classes = np.unique(y_true)
    
    for _ in range(n_bootstraps):
        indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        if multi_class:
            y_true_bin = label_binarize(y_true[indices], classes=unique_classes)
            score = metric_func(y_true_bin, y_pred_proba[indices], **metric_kwargs)
        else:
            score = metric_func(y_true[indices], y_pred_label[indices], **metric_kwargs)
        scores.append(score)
    scores = np.array(scores)
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(scores)
    return mean_score, lower, upper

# 模型评估及置信区间计算
performance_results = []
class_performance_results = []

unique_classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=unique_classes)

def bootstrap_ci_binary(y_true, y_pred_label=None, y_pred_prob=None, metric_func=None,
                        n_bootstraps=1000, alpha=0.05, **metric_kwargs):
    """
    针对单个类别的二分类指标计算bootstrap置信区间
    :param y_true: 二分类真实标签 (0/1)
    :param y_pred_label: 二分类预测标签 (0/1)
    :param y_pred_prob: 二分类预测概率 (概率得分)
    :param metric_func: 计算指标的函数，传入y_true及相应参数
    :param n_bootstraps: bootstrap次数
    :param alpha: 显著水平，通常0.05
    :param metric_kwargs: 传入metric_func的其他参数
    :return: (均值, 下限, 上限)
    """
    rng = np.random.RandomState(42)
    scores = []
    y_true = np.array(y_true)
    if y_pred_label is not None:
        y_pred_label = np.array(y_pred_label)
    if y_pred_prob is not None:
        y_pred_prob = np.array(y_pred_prob)

    n_samples = len(y_true)

    for i in range(n_bootstraps):
        indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        yt_sample = y_true[indices]

        if metric_func == roc_auc_score:
            # ROC AUC用概率
            yp_sample = y_pred_prob[indices]
        else:
            # 其他根据是否有预测标签选择
            yp_sample = y_pred_label[indices] if y_pred_label is not None else y_pred_prob[indices]

        try:
            score = metric_func(yt_sample, yp_sample, **metric_kwargs)
        except ValueError:
            # 部分bootstrap样本标签单类时指标无法计算，跳过
            continue
        scores.append(score)

    scores = np.array(scores)
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(scores)

    return mean_score, lower, upper

print("开始计算各模型性能及置信区间...")
for model_name, model in best_models.items():
    print(f"评估 {model_name} ...")
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)
    
    macro_f1_mean, macro_f1_low, macro_f1_high = bootstrap_ci(
        y_test, y_pred, y_pred_proba, f1_score, n_bootstraps=1000, alpha=0.05, multi_class=False, average='macro'
    )
    accuracy_mean, accuracy_low, accuracy_high = bootstrap_ci(
        y_test, y_pred, y_pred_proba, accuracy_score, n_bootstraps=1000, alpha=0.05, multi_class=False
    )
    precision_mean, precision_low, precision_high = bootstrap_ci(
        y_test, y_pred, y_pred_proba, precision_score, n_bootstraps=1000, alpha=0.05, multi_class=False, average='macro'
    )
    recall_mean, recall_low, recall_high = bootstrap_ci(
        y_test, y_pred, y_pred_proba, recall_score, n_bootstraps=1000, alpha=0.05, multi_class=False, average='macro'
    )
    roc_auc_mean, roc_auc_low, roc_auc_high = bootstrap_ci(
        y_test, y_pred, y_pred_proba,
        lambda yt, yp: roc_auc_score(yt, yp, multi_class='ovr', average='macro'),
        n_bootstraps=1000, alpha=0.05, multi_class=True
    )
    
    performance_results.append({
        "Model": model_name,
        "Macro F1 Score": f"{macro_f1_mean:.3f} ({macro_f1_low:.3f}, {macro_f1_high:.3f})",
        "Accuracy": f"{accuracy_mean:.3f} ({accuracy_low:.3f}, {accuracy_high:.3f})",
        "Macro Precision": f"{precision_mean:.3f} ({precision_low:.3f}, {precision_high:.3f})",
        "Macro Recall": f"{recall_mean:.3f} ({recall_low:.3f}, {recall_high:.3f})",
        "Macro ROC AUC": f"{roc_auc_mean:.3f} ({roc_auc_low:.3f}, {roc_auc_high:.3f})"
    })
    
    # 分类别指标，不计算置信区间
    class_report = classification_report(y_test, y_pred, target_names=[str(c) for c in unique_classes], output_dict=True)
    for i, cls in enumerate(unique_classes):
        y_test_cls = (y_test == cls).astype(int)
        y_pred_cls_label = (y_pred == cls).astype(int)
        y_pred_cls_prob = y_pred_proba[:, i]

        cls_acc_mean, cls_acc_low, cls_acc_high = bootstrap_ci_binary(
            y_test_cls, y_pred_label=y_pred_cls_label, metric_func=accuracy_score,
            n_bootstraps=1000, alpha=0.05
        )
        cls_f1_mean, cls_f1_low, cls_f1_high = bootstrap_ci_binary(
            y_test_cls, y_pred_label=y_pred_cls_label, metric_func=f1_score,
            n_bootstraps=1000, alpha=0.05
        )
        cls_prec_mean, cls_prec_low, cls_prec_high = bootstrap_ci_binary(
            y_test_cls, y_pred_label=y_pred_cls_label, metric_func=precision_score,
            n_bootstraps=1000, alpha=0.05
        )
        cls_rec_mean, cls_rec_low, cls_rec_high = bootstrap_ci_binary(
            y_test_cls, y_pred_label=y_pred_cls_label, metric_func=recall_score,
            n_bootstraps=1000, alpha=0.05
        )
        cls_auc_mean, cls_auc_low, cls_auc_high = bootstrap_ci_binary(
            y_test_cls, y_pred_prob=y_pred_cls_prob, metric_func=roc_auc_score,
            n_bootstraps=1000, alpha=0.05
        )

        support = (y_test_cls == 1).sum()

        class_performance_results.append({
            "Model": model_name,
            "Class": cls,
            "Accuracy": f"{cls_acc_mean:.3f} ({cls_acc_low:.3f}, {cls_acc_high:.3f})",
            "F1 Score": f"{cls_f1_mean:.3f} ({cls_f1_low:.3f}, {cls_f1_high:.3f})",
            "Precision": f"{cls_prec_mean:.3f} ({cls_prec_low:.3f}, {cls_prec_high:.3f})",
            "Recall": f"{cls_rec_mean:.3f} ({cls_rec_low:.3f}, {cls_rec_high:.3f})",
            "AUC": f"{cls_auc_mean:.3f} ({cls_auc_low:.3f}, {cls_auc_high:.3f})",
            "Support": support
        })

performance_df = pd.DataFrame(performance_results)
class_performance_df = pd.DataFrame(class_performance_results)

print("模型性能评估（含95%置信区间）：")
print(performance_df)

performance_df.to_excel(r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\BM-model_performance_SVI_CI.xlsx", index=False)
class_performance_df.to_excel(r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\class_performance_SVI_CI.xlsx", index=False)

# 绘制混淆矩阵示例（以最后一个模型为例）
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f'Confusion Matrix: {model_name}')
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# 确保y_test和best_models是之前代码中定义的
# X_test_processed是经过数值标准化和类别特征OneHot编码后的测试集

unique_classes = np.unique(y_test)
n_classes = len(unique_classes)
y_test_bin = label_binarize(y_test, classes=unique_classes)

X_test_input = X_test_processed  # 经过预处理的测试集

n_bootstraps = 1000
rng_seed = 42
rng = np.random.RandomState(rng_seed)

plt.figure(figsize=(12, 8))
colors = ['#297fb8', '#f0c514', '#23a54f', '#f373e4', '#6e3383', '#cd6155', '#43735d', '#f97506']

for (model_name, model), color in zip(best_models.items(), colors):
    # 预测概率
    y_pred_proba = model.predict_proba(X_test_input)
    
    # 计算每类FPR、TPR和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 所有类别FPR的全集合
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # 对每个类别的TPR按all_fpr插值求平均，得到macro平均TPR
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    # 计算macro平均AUC
    macro_roc_auc = auc(all_fpr, mean_tpr)
    
    # Bootstrap估计95%置信区间
    bootstrapped_scores = []
    n_samples = y_test_bin.shape[0]
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_test_bin[indices], axis=0)) < n_classes:
            # 跳过不包含全部类别的bootstrap样本
            continue
        
        y_true_boot = y_test_bin[indices]
        y_pred_boot = y_pred_proba[indices]
        
        fpr_boot = dict()
        tpr_boot = dict()
        for i in range(n_classes):
            fpr_boot[i], tpr_boot[i], _ = roc_curve(y_true_boot[:, i], y_pred_boot[:, i])
        all_fpr_boot = np.unique(np.concatenate([fpr_boot[i] for i in range(n_classes)]))
        mean_tpr_boot = np.zeros_like(all_fpr_boot)
        for i in range(n_classes):
            mean_tpr_boot += np.interp(all_fpr_boot, fpr_boot[i], tpr_boot[i])
        mean_tpr_boot /= n_classes
        score = auc(all_fpr_boot, mean_tpr_boot)
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    plt.plot(
        all_fpr,
        mean_tpr,
        color=color,
        label=f'{model_name} (Macro AUC = {macro_roc_auc:.2f} [{ci_lower:.2f}, {ci_upper:.2f}])',
        lw=2,
        alpha=0.8
    )

plt.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d', alpha=0.6, lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Macro-Averaged ROC Curves of All Models',fontsize=20)
plt.legend(loc='lower right',fontsize=14)
plt.grid(False)

output_path = r"C:\Users\ACER\Desktop\test\stt\picture\BM\BM_Model_ROC.jpg"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

category_labels = ['very high-risk', 'high-risk', 'low-risk']
cmap_hex_dict = {
    "Logistic Regression": ['#d4e6f1', '#2980b9'],
    "SVM": ['#f9e79f', '#f1c40f'],
    "Random Forest": ['#B9EACA', '#23A44E'],
    "K-Nearest Neighbors": ['#F6B9E4', '#F373E4'],
    "Naive Bayes": ['#d7bde2', '#6c3483'],
    "AdaBoost": ['#fadbd8', '#cd6155'],
    "Gradient Boosting": ['#AFDBDF', '#43735C'],
    "CatBoost": ['#F3C6A0', '#F97505']
}

output_cm_dir = r"C:\Users\ACER\Desktop\test\stt\picture\BM\confusion_matrices"
os.makedirs(output_cm_dir, exist_ok=True)

unique_classes = np.unique(y)  # 确保y是全部样本标签

# 确认unique_classes和category_labels长度一致及对应
assert len(unique_classes) == len(category_labels), "unique_classes和category_labels长度不匹配！"

for model_name, model in best_models.items():
    y_pred = model.predict(X_test_processed)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    # 这里用category_labels替代unique_classes作为显示标签
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_labels)
    
    plt.figure(figsize=(12, 8))
    
    hex_colors = cmap_hex_dict.get(model_name, ['#e0e0e0', '#000000'])
    custom_cmap = LinearSegmentedColormap.from_list(f'{model_name}_cmap', hex_colors)
    
    ax = plt.gca()
    disp.plot(cmap=custom_cmap, values_format='d', ax=ax)
    
    # 隐藏 sklearn 自动绘制的数字文本
    for txt in ax.texts:
        txt.set_visible(False)
    
    im = ax.images[0]  # 热力图对象
    
    plt.title(f"{model_name}", fontsize=20, fontweight='bold')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    col_sum = cm.sum(axis=1, keepdims=True)  # 预测类别总数
    col_sum_safe = np.where(col_sum == 0, 1, col_sum)  # 避免除零错误
    cm_percent = cm / col_sum_safe * 100
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            text = f"{count}\n{percent:.2f}%"
            
            rgba = im.cmap(im.norm(count))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = 'black' if brightness > 0.5 else 'white'
            
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=26)
    
    plt.tight_layout()
    plt.xlabel('Predicted',fontsize=26)
    plt.ylabel('True',fontsize=26)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    save_path = os.path.join(output_cm_dir, f"{model_name} confusion matrix.jpg")
    plt.savefig(save_path, dpi=600, format='jpg', bbox_inches='tight')
    print(f"保存混淆矩阵图像：{save_path}")
    plt.show()


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 初始化结果列表
results = []

for model_name, model in best_models.items():
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)  # 用于AUC
    
    # 针对每个类别计算指标
    for cls in unique_classes:
        # 该类别的二分类标签（是否为该类）
        y_test_bin = (y_test == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        acc = accuracy_score(y_test_bin, y_pred_bin)
        prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)

        # 计算AUC需要预测概率，保证类别索引正确
        try:
            cls_idx = list(model.classes_).index(cls)
            auc_score = roc_auc_score(y_test_bin, y_pred_proba[:, cls_idx])
        except Exception:
            auc_score = np.nan  # 若计算失败则NaN
        
        results.append({
            'Model': model_name,
            'Class': cls,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'AUC': auc_score
        })

# 转成DataFrame
df_metrics = pd.DataFrame(results)

# 若需要，保存成Excel以供后续绘图或复核
excel_save_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\class-performance-generated.xlsx"
df_metrics.to_excel(excel_save_path, index=False)
print(f"性能指标已保存至：{excel_save_path}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# 1. 读取Excel数据
file_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\class-performance-generated.xlsx"
df = pd.read_excel(file_path)

# 2. 指标名称和顺序，保持和雷达图绘制顺序一致
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']

# 3. 雷达图绘制函数，支持标题文字颜色和圆角矩形边框颜色分别设置，图例颜色自定义
def plot_radar(metrics, class_labels, metric_labels, legend_names=None, colors=None,
               legend_loc='upper center', title_text="Clinical Model",
               title_bg_width=1.0, title_bg_linewidth=1.5,
               title_bg_color='gray', title_text_color=None,
               title_y=1.0, fontsize=14, output_path=None):

    if title_text_color is None:
        title_text_color = title_bg_color  # 默认文字颜色和矩形边框颜色相同

    num_vars = len(metric_labels)
    sides = 5
    pentagon_angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.pi / 2

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
        vals = row[:sides].tolist()
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
    ax.set_xticks([])

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

    # 添加径向数值刻度文本（顶部角度方向）
    r_ticks = grid_radii
    r_tick_labels = [f"{r:.1f}" for r in r_ticks]
    angle_for_labels = np.pi / 2

    for r, label in zip(r_ticks, r_tick_labels):
        ax.text(angle_for_labels, r, label,
                fontsize=20,
                color='gray',
                horizontalalignment='center',
                verticalalignment='bottom',
                rotation=0,
                rotation_mode='anchor')

    # 添加径向网格线（中心到各顶点）
    for angle in pentagon_angles:
        ax.plot([angle, angle], [0, 1], color='gray', linestyle='dotted', linewidth=1, zorder=0)

    # 角度标签稍超出边界
    label_r = 1.2
    for angle, label in zip(pentagon_angles, display_labels):
        ax.text(angle, label_r, label, fontsize=26,
                horizontalalignment='center',
                verticalalignment='center')

    # 添加图例
    legend = ax.legend(
        loc=legend_loc,
        bbox_to_anchor=(0.5, 1.25),
        ncol=len(class_labels),
        fontsize=24,
        frameon=False
    )
    for text, color in zip(legend.get_texts(), colors):
        text.set_color(color)

    # 添加标题及圆角矩形
    title = fig.suptitle(title_text, fontsize=28, y=title_y, color=title_text_color,fontweight='bold')

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

    if output_path:
        plt.savefig(output_path, dpi=600, format='jpg', bbox_inches='tight')
        print(f"保存图片：{output_path}")
    plt.show()


# 4. 类别和图例名称处理
all_classes = df['Class'].unique()

custom_legend_dict = {
    0: 'very high-risk',
    1: 'high-risk',
    2: 'low-risk',
    # 需要时可以继续补充
}
legend_names_all = [custom_legend_dict.get(cls, str(cls)) for cls in all_classes]

# 5. 颜色配置示例
base_colors = ['#e74c3c', '#3498db', '#f39c12']
if len(all_classes) > len(base_colors):
    cmap = plt.cm.get_cmap('tab10', len(all_classes))
    colors_all_default = [cmap(i) for i in range(len(all_classes))]
else:
    colors_all_default = base_colors[:len(all_classes)]

# 不同模型对应的图例颜色（示例，按需定义）
model_colors_dict = {
    'Logistic Regression': ['#c5dcec', '#76aed1', '#297fb8'],
    'SVM': ['#f8e598', '#f5d75b', '#f0c514'],
    'Random Forest': ['#abe3be', '#6dc68e', '#23a54f'],
    'K-Nearest Neighbors': ['#f6b9e3','#f59be3','#f373e4'],
    'Naive Bayes':['#ccb0d9','#9d71ae','#6e3383'],
    'AdaBoost':['#f3cac6','#e39c96','#cd6155'],
    'Gradient Boosting':['#afdbde','#76a499','#43735d'],
    'CatBoost':['#f3c69f','#f79d51','#f97506']
    # 这里请替换为您的实际模型名和颜色列表
}

# 不同模型标题颜色（示例）
model_title_colors = {
    'Logistic Regression': '#297fb8',
    'SVM': '#f0c514',
    'Random Forest': '#23a54f',
    'K-Nearest Neighbors':'#f373e4',
    'Naive Bayes':'#6e3383',
    'AdaBoost':'#cd6155',
    'Gradient Boosting':'#43735d',
    'CatBoost':'#f97506'
}

# 6. 输出目录
output_dir = r"C:\Users\ACER\Desktop\test\stt\picture\BM\radar"
os.makedirs(output_dir, exist_ok=True)

# 7. 按模型绘图
models = df['Model'].unique()
for model in models:
    df_model = df[df['Model'] == model]
    df_model = df_model.set_index('Class').reindex(all_classes)
    metrics_mat = df_model[metrics_names].to_numpy()

    # 取颜色，默认fallback到默认颜色
    colors_this = model_colors_dict.get(model, colors_all_default)
    title_color = model_title_colors.get(model, '#16a085')  # 默认绿色

    output_file = os.path.join(output_dir, f"{model}_radar.jpg")

    plot_radar(
        metrics=metrics_mat,
        class_labels=all_classes,
        metric_labels=metrics_names,
        legend_names=legend_names_all,
        colors=colors_this,
        legend_loc='upper center',
        title_text=f"{model}",
        title_bg_width=0.6,
        title_bg_linewidth=2,
        title_bg_color=title_color,     # 圆角矩形边框颜色
        title_text_color=title_color,   # 标题文字颜色
        title_y=1.03,
        output_path=output_file
    )


import numpy as np
import matplotlib.pyplot as plt

def plot_decision_curve_classic(y_true, y_pred_proba, models, unique_classes, titles=None):
    total_samples = len(y_true)
    n_classes = len(unique_classes)
    plt.figure(figsize=(5 * (n_classes + 1), 5))  # 多加一列用于宏DCA曲线

    if titles is None:
        titles = [f'Class {cls}' for cls in unique_classes]
    elif len(titles) != n_classes:
        raise ValueError("参数 titles 长度应与 unique_classes 长度一致")

    thresholds = np.linspace(0.01, 0.99, 100)  # 避免阈值为0或1导致计算异常

    # 用于存储宏平均净收益，初始为0数组
    macro_net_benefits = {model_name: np.zeros_like(thresholds) for model_name in models.keys()}

    for cls_idx, cls in enumerate(unique_classes):
        plt.subplot(1, n_classes + 1, cls_idx + 1)

        net_benefits = {}
        pos_total = np.sum(y_true == cls)

        for model_name in models.keys():
            probs = y_pred_proba[model_name][:, cls_idx]
            net_benefit = []

            for pt in thresholds:
                y_pred_threshold = (probs >= pt).astype(int)

                tp = np.sum((y_true == cls) & (y_pred_threshold == 1))
                fp = np.sum((y_true != cls) & (y_pred_threshold == 1))

                net_ben = (tp / total_samples) - (fp / total_samples) * (pt / (1 - pt))
                net_benefit.append(net_ben)

            net_benefits[model_name] = net_benefit
            # 累加到宏DCA净收益，用于后续平均
            macro_net_benefits[model_name] += np.array(net_benefit)

        # 绘制各模型该类别净收益曲线
        for model_name, benefits in net_benefits.items():
            plt.plot(thresholds, benefits, lw=2, label=model_name)

        # Treat All曲线
        prevalence = pos_total / total_samples if total_samples > 0 else 0
        treat_all_net_benefit = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
        plt.plot(thresholds, treat_all_net_benefit, color='black', linestyle='--', lw=2, label='Treat All')

        # Treat None线
        plt.axhline(0, color='gray', linestyle='--', lw=2, label='Treat None')

        plt.title(titles[cls_idx],fontsize=18)
        plt.xlabel('Threshold Probability',fontsize=18)
        plt.ylabel('Net Benefit',fontsize=18)
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.legend(fontsize=12)
        plt.ylim(bottom=-0.05, top=0.4)

    # 绘制宏平均DCA曲线（所有类别净收益均值）
    plt.subplot(1, n_classes + 1, n_classes + 1)

    for model_name in models.keys():
        avg_net_benefit = macro_net_benefits[model_name] / n_classes  # 取平均
        plt.plot(thresholds, avg_net_benefit, lw=2, label=model_name)

    # 宏平均Treat All曲线（各类别患病率均值）
    prevalence_all = np.mean([np.sum(y_true == cls) / total_samples for cls in unique_classes])
    treat_all_macro = [prevalence_all - (1 - prevalence_all) * (pt / (1 - pt)) for pt in thresholds]
    plt.plot(thresholds, treat_all_macro, color='black', linestyle='--', lw=2, label='Treat All')

    # Treat None线
    plt.axhline(0, color='gray', linestyle='--', lw=2, label='Treat None')

    plt.title('Macro Average DCA',fontsize=18)
    plt.xlabel('Threshold Probability',fontsize=18)
    plt.ylabel('Net Benefit',fontsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend(fontsize=12)
    plt.ylim(bottom=-0.05, top=0.4)

    plt.tight_layout()
    output_path = r"C:\Users\ACER\Desktop\test\stt\picture\BM\DCA.jpg"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()


# 以下是调用示例，确保best_models、X_test_processed、y_test、unique_classes均已定义且正确
y_pred_probas = {
    model_name: model.predict_proba(X_test_processed)
    for model_name, model in best_models.items()
 }
titles = ['Decision Curve Analysis(very high-risk)', 'Decision Curve Analysis(high-risk)', 'Decision Curve Analysis(low-risk)']
plot_decision_curve_classic(y_test, y_pred_probas, best_models, unique_classes, titles=titles)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve(y_true, y_pred_proba, models, unique_classes, titles=None):
    plt.figure(figsize=(5 * (len(unique_classes) + 1), 5))  # 多加一列画宏平均PR曲线

    if titles is None:
        titles = [f'Class {cls}' for cls in unique_classes]
    elif len(titles) != len(unique_classes):
        raise ValueError("参数 titles 长度应与 unique_classes 长度一致")

    for cls_idx, cls in enumerate(unique_classes):
        plt.subplot(1, len(unique_classes) + 1, cls_idx + 1)

        for model_name in models.keys():
            probs = y_pred_proba[model_name][:, cls_idx]
            y_true_binary = (y_true == cls).astype(int)

            precision, recall, _ = precision_recall_curve(y_true_binary, probs)
            avg_precision = average_precision_score(y_true_binary, probs)

            plt.plot(recall, precision, label=f'{model_name} (AP={avg_precision:.2f})')

        plt.title(titles[cls_idx])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(bbox_to_anchor=(0.99, 0.01), loc='lower right')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

    # 宏平均PR曲线绘制：micro-average（所有类别样本合并计算）
    plt.subplot(1, len(unique_classes) + 1, len(unique_classes) + 1)
    for model_name in models.keys():
        # 合并所有类别的真实标签和预测概率（micro平均）
        # y_true_multi: shape (n_samples,)
        # y_pred_proba_multi: shape (n_samples, n_classes)
        # 计算micro-average时需要将多分类转为二分类形式，采用OneVsRest策略
        # sklearn的average_precision_score支持multi_class='ovr'参数，但precision_recall_curve不支持多类，需手动转换

        # 先构造多标签二值矩阵
        y_true_bin = np.zeros((len(y_true), len(unique_classes)))
        for i, c in enumerate(unique_classes):
            y_true_bin[:, i] = (y_true == c).astype(int)

        # 计算micro average的precision-recall
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_proba[model_name].ravel())
        avg_precision = average_precision_score(y_true_bin, y_pred_proba[model_name], average='micro')

        plt.plot(recall, precision, label=f'{model_name} (micro-avg AP={avg_precision:.2f})')

    plt.title('Macro Average PR Curve (Micro-Average)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(bbox_to_anchor=(0.99, 0.01), loc='lower right')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.tight_layout()
    output_path = r"C:\Users\ACER\Desktop\test\stt\picture\BM\PR-Curve.jpg"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()


# 调用示例，确保best_models、X_test_processed、y_test、unique_classes已定义
y_pred_probas = {
    model_name: model.predict_proba(X_test_processed)
    for model_name, model in best_models.items()
}

unique_classes = np.unique(y_test)
titles = ['Precision-Recall Curve(< 1)', 'Precision-Recall Curve(1--4)', 'Precision-Recall Curve(> 4)']  # 与类别数对应

plot_precision_recall_curve(y_test, y_pred_probas, best_models, unique_classes, titles=titles)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
import joblib

# 假设您已定义best_models字典，含对应训练好的基学习器对象
# 示例：
# best_models = {
#     "Logistic Regression": lr_model,
#     "Random Forest": rf_model,
#     "CatBoost": catboost_model
# }

estimators = [
    ('lr', best_models["Logistic Regression"]),
    ('rf', best_models["Random Forest"]),
    ('adaboost', best_models["AdaBoost"])
]

X_train = X_train_resampled
y_train = y_train_resampled
X_test = X_test_processed
y_test = y_test
unique_classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=unique_classes)

random_seed = 42  # 请根据实际设置

def stacking_grid_search(meta_learner, param_grid, meta_name, save_path=None):
    print(f"开始对Stacking模型（元学习器：{meta_name}）进行网格搜索...")

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    grid_search = GridSearchCV(
        stacking_clf,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"{meta_name} 元学习器最优参数: {grid_search.best_params_}")
    print(f"最佳交叉验证宏平均F1: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    macro_roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')

    print(f"{meta_name} 元学习器Stacking模型测试集性能：")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro ROC AUC: {macro_roc_auc:.4f}")

    # 计算每个类别的评价指标
    class_precisions = precision_score(y_test, y_pred, average=None, labels=unique_classes)
    class_recalls = recall_score(y_test, y_pred, average=None, labels=unique_classes)
    class_f1s = f1_score(y_test, y_pred, average=None, labels=unique_classes)

    summary_df = pd.DataFrame({
        'Metric': ['Macro F1', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro ROC AUC'],
        'Value': [macro_f1, accuracy, precision, recall, macro_roc_auc]
    })

    per_class_df = pd.DataFrame({
        'Class': unique_classes,
        'Precision': class_precisions,
        'Recall': class_recalls,
        'F1-Score': class_f1s
    })

    params_df = pd.DataFrame(list(grid_search.best_params_.items()), columns=['Parameter', 'Best Value'])

    # 混淆矩阵绘制
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Stacking Confusion Matrix ({meta_name})')
    plt.tight_layout()
    plt.show()

    return best_model, summary_df, per_class_df, params_df


meta_learners = {
    "RandomForest": (
        RandomForestClassifier(random_state=random_seed),
        {
            'final_estimator__n_estimators': [50, 100],
            'final_estimator__max_depth': [None, 10, 20],
            'final_estimator__min_samples_split': [2, 5]
        }
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=5000, random_state=random_seed),
        {
            'final_estimator__C': [0.01, 0.1, 1, 10],
            'final_estimator__penalty': ['l2'],
            'final_estimator__solver': ['lbfgs', 'liblinear']
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=random_seed),
        {
            'final_estimator__n_estimators': [50, 100],
            'final_estimator__learning_rate': [0.01, 0.1, 1]
        }
    )
}

excel_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\stacking_models_evaluation.xlsx"

with pd.ExcelWriter(excel_path, engine='openpyxl') as excel_writer:
    best_stacking_models = {}

    for name, (meta_learner, param_grid) in meta_learners.items():
        print(f"\n==== 开始处理元学习器：{name} ====\n")
        try:
            save_fig_path = excel_path.replace('.xlsx', f'_{name}_confusion.jpg')
            best_model, summary_df, per_class_df, params_df = stacking_grid_search(
                meta_learner, param_grid, name, save_path=save_fig_path
            )
            best_stacking_models[name] = best_model

            # 保存完整Stacking模型，只保存逻辑回归元学习器对应的模型
            if name == "LogisticRegression":
                save_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\stacking_logistic_regression_model.pkl"
                joblib.dump(best_model, save_path)
                print(f"完整Stacking模型（含基学习器和元学习器）已保存到: {save_path}")
                scaler_save_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\scaler.pkl"
                joblib.dump(scaler, scaler_save_path)
                print(f"Scaler已保存到: {scaler_save_path}")

            # 写入Excel
            summary_df.to_excel(excel_writer, sheet_name=f'{name}_Summary', index=False)
            per_class_df.to_excel(excel_writer, sheet_name=f'{name}_PerClass', index=False)
            if not params_df.empty:
                params_df.to_excel(excel_writer, sheet_name=f'{name}_Params', index=False)
            else:
                print(f"{name} 的 params_df 为空，跳过写入。")

        except Exception as e:
            print(f"处理元学习器 {name} 时发生错误: {e}")

print(f"\n所有元学习器结果已成功保存至Excel: {excel_path}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import pandas as pd

def bootstrap_auc(y_true_bin, y_score_cls, n_bootstrap=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    aucs = []
    n_samples = len(y_true_bin)
    for _ in range(n_bootstrap):
        indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        y_true_sample = y_true_bin.iloc[indices]
        y_score_sample = y_score_cls.iloc[indices]

        if len(np.unique(y_true_sample)) < 2:
            continue
        score = roc_auc_score(y_true_sample, y_score_sample)
        aucs.append(score)
    return np.array(aucs)

def plot_roc_multiclass_with_macro_in_subplot(meta_lr_model, X_test, y_test, unique_classes,
                                              n_bootstrap=1000, alpha=0.95, save_path=None,
                                              class_colors=None, macro_color='black'):
    """
    绘制多分类ROC及宏平均ROC，带AUC及95%置信区间，支持自定义曲线颜色。

    参数：
    - meta_lr_model: 训练好的多分类模型，需实现predict_proba()
    - X_test: 测试特征
    - y_test: 测试标签
    - unique_classes: 类别列表
    - n_bootstrap: bootstrap采样次数
    - alpha: 置信水平
    - save_path: 保存路径，None则不保存
    - class_colors: list或tuple，类别曲线颜色列表，长度一般等于类别数，默认None表示全部用'darkorange'
    - macro_color: 宏平均曲线颜色，默认'black'
    """
    y_score = meta_lr_model.predict_proba(X_test)
    n_classes = len(unique_classes)

    if class_colors is None or len(class_colors) < n_classes:
        # 若未传颜色或长度不足，全部默认橙色
        class_colors = ['darkorange'] * n_classes

    # 预备宏平均ROC曲线需要的fpr集合
    all_fpr = np.unique(np.concatenate([
        roc_curve((y_test == cls).astype(int), y_score[:, i])[0]
        for i, cls in enumerate(unique_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)

    auc_list = []
    auc_ci_list = []
    bootstrapped_aucs_all = []

    titles = ['Meta Model ROC(< 1)', 'Meta Model ROC(1--4)', 'Meta Model ROC(> 4)']  # 根据实际类别适当修改

    plt.figure(figsize=(6 * (n_classes + 1), 5))  # 多一张宏平均

    for i, cls in enumerate(unique_classes):
        plt.subplot(1, n_classes + 1, i + 1)

        y_test_bin = (y_test == cls).astype(int)
        y_score_cls = y_score[:, i]

        fpr, tpr, _ = roc_curve(y_test_bin, y_score_cls)
        roc_auc = auc(fpr, tpr)

        y_test_bin_series = pd.Series(y_test_bin).reset_index(drop=True)
        y_score_cls_series = pd.Series(y_score_cls).reset_index(drop=True)

        aucs = bootstrap_auc(y_test_bin_series, y_score_cls_series,
                             n_bootstrap=n_bootstrap, random_state=42 + i)
        if len(aucs) == 0:
            print(f"类别 {cls} bootstrap样本不足，AUC置信区间计算失败")
            ci_lower, ci_upper = np.nan, np.nan
        else:
            ci_lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
            ci_upper = np.percentile(aucs, (1 + alpha) / 2 * 100)

        auc_list.append(roc_auc)
        auc_ci_list.append((ci_lower, ci_upper))
        bootstrapped_aucs_all.append(aucs)

        mean_tpr += np.interp(all_fpr, fpr, tpr)

        plt.plot(fpr, tpr, color=class_colors[i],
                 lw=2, label=f'AUC = {roc_auc:.2f}[{ci_lower:.2f}, {ci_upper:.2f}]')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate',fontsize=18)
        plt.ylabel('True Positive Rate',fontsize=18)
        title = titles[i] if i < len(titles) else f'Class {cls}'
        plt.title(title,fontsize=20)
        plt.legend(loc="lower right",fontsize=16)
        plt.grid(False)

    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    min_bootstrap_len = min(len(a) for a in bootstrapped_aucs_all)
    aligned_aucs = np.array([a[:min_bootstrap_len] for a in bootstrapped_aucs_all]).T
    macro_bootstrap_aucs = aligned_aucs.mean(axis=1)
    if len(macro_bootstrap_aucs) == 0:
        macro_ci_lower, macro_ci_upper = np.nan, np.nan
        print("宏平均AUC bootstrap样本不足，置信区间计算失败")
    else:
        macro_ci_lower = np.percentile(macro_bootstrap_aucs, (1 - alpha) / 2 * 100)
        macro_ci_upper = np.percentile(macro_bootstrap_aucs, (1 + alpha) / 2 * 100)

    plt.subplot(1, n_classes + 1, n_classes + 1)
    plt.plot(all_fpr, mean_tpr, color=macro_color, lw=2, linestyle='-',
             label=f'AUC={macro_auc:.2f}[{macro_ci_lower:.2f},{macro_ci_upper:.2f}]')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('Meta Model Macro-average ROC',fontsize=20)
    plt.legend(loc="lower right",fontsize=16)
    plt.grid(False)

    plt.tight_layout()

   # if save_path:
        #plt.savefig(save_path, dpi=600, bbox_inches='tight')
       # print(f"ROC曲线图已保存至: {save_path}")

    plt.show()

    return {
        'per_class_auc': auc_list,
        'per_class_ci': auc_ci_list,
        'macro_auc': macro_auc,
        'macro_ci': (macro_ci_lower, macro_ci_upper)
    }

colors = ['#AA0A17', '#AA0A17', '#AA0A17']  # 自定义3个类别颜色，比如
macro_curve_color = '#AA0A17'       # 宏平均ROC曲线颜色

result = plot_roc_multiclass_with_macro_in_subplot(
    meta_lr_model=best_stacking_models["LogisticRegression"],
    X_test=X_test_processed,
    y_test=y_test,
    unique_classes=unique_classes,
    n_bootstrap=1000,
    alpha=0.95,
    #save_path=r"C:\Users\ACER\Desktop\test\stt\picture\BM\stack-LR-roc.jpg",
    class_colors=colors,
    macro_color=macro_curve_color
)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 类别映射标签
category_labels = ['< 1', '1--4', '> 4']

# 预定义颜色映射（示例）
cmap_hex_dict = {
    "Logistic Regression": ['#FEE0E2', '#AA0A17']
}

output_cm_dir = r"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking-confusion_matrices"
os.makedirs(output_cm_dir, exist_ok=True)

# 确保所有标签类别一致
unique_classes = np.unique(y)  # 这里的y是整个数据标签

assert len(unique_classes) == len(category_labels), "unique_classes和category_labels长度不匹配！"

# 逻辑回归元学习器变量名，替换成您实际的模型变量
model_name = "Logistic Regression"
model = best_stacking_models["LogisticRegression"]  # 您训练好的逻辑回归元学习器

# 预测测试集
y_pred = model.predict(X_test_processed)

# 计算混淆矩阵，labels使用unique_classes保证顺序一致
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

# 混淆矩阵可视化，显示自定义类别标签
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_labels)

plt.figure(figsize=(12, 8))

# 取对应模型的颜色，不存在则默认灰色渐变
hex_colors = cmap_hex_dict.get(model_name, ['#e0e0e0', '#000000'])
custom_cmap = LinearSegmentedColormap.from_list(f'{model_name}_cmap', hex_colors)

ax = plt.gca()
disp.plot(cmap=custom_cmap, values_format='d', ax=ax)

# 隐藏 sklearn 自动绘制的数字文本，方便自定义显示数字和百分比
for txt in ax.texts:
    txt.set_visible(False)

im = ax.images[0]  # 获取热力图对象

plt.title("Stacking Model", fontsize=24, fontweight='bold')
plt.xlabel("Predicted label")
plt.ylabel("True label")

# 计算百分比
col_sum = cm.sum(axis=1, keepdims=True)  # 预测类别总数
col_sum_safe = np.where(col_sum == 0, 1, col_sum)  # 避免除零错误
cm_percent = cm / col_sum_safe * 100

# 自定义文本显示格式（数量 + 百分比）及字体颜色（根据背景亮度自动调整）
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percent = cm_percent[i, j]
        text = f"{count}\n{percent:.2f}%"

        rgba = im.cmap(im.norm(count))
        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        color = 'black' if brightness > 0.5 else 'white'

        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=26)

plt.tight_layout()
plt.xlabel('Predicted',fontsize=22)
plt.ylabel('True',fontsize=22)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
save_path = os.path.join(output_cm_dir, f"{model_name} confusion matrix.jpg")
plt.savefig(save_path, dpi=600, format='jpg', bbox_inches='tight')
print(f"保存混淆矩阵图像：{save_path}")

plt.show()


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize

def plot_logreg_calibration_curve(
        model,
        X_train, y_train,
        X_test, y_test,
        classes=None,
        class_names=None,
        colors=None,
        n_bins=15
    ):
    """
    只绘制逻辑回归元学习器的多分类校准曲线，训练集和测试集分别绘制。
    
    参数：
    - model：训练好的逻辑回归元学习器堆叠模型
    - X_train, y_train：训练集特征和标签
    - X_test, y_test：测试集特征和标签
    - classes：类别列表，如 None 则自动获取
    - class_names：类别对应显示名称列表，长度应与classes一致
    - colors：类别线条颜色列表，长度应与classes一致
    - n_bins：校准曲线分箱数，默认15
    """

    if classes is None:
        classes = np.unique(y_train)
    n_classes = len(classes)

    if class_names is None:
        class_names = [f'Class {c}' for c in classes]
    if colors is None:
        # 默认颜色列表，长度不足时循环使用
        colors = plt.cm.tab10.colors
        if n_classes > len(colors):
            colors = colors * (n_classes // len(colors) + 1)
        colors = colors[:n_classes]

    # 标签二值化
    y_train_bin = label_binarize(y_train, classes=classes)
    y_test_bin = label_binarize(y_test, classes=classes)

    # 预测概率
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 训练集校准曲线
    ax = axes[0]
    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(y_train_bin[:, i], y_train_proba[:, i], n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker='o', label=class_names[i], color=colors[i])
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_title('Calibration Curve - Training Set')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.legend(loc='best')
    ax.grid(False)

    # 测试集校准曲线
    ax = axes[1]
    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(y_test_bin[:, i], y_test_proba[:, i], n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker='o', label=class_names[i], color=colors[i])
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_title('Calibration Curve - Test Set')
    ax.set_xlabel('Predicted Probability')
    ax.legend(loc='best')
    ax.grid(False)
    #plt.savefig(r"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking_calibration_curves.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# 自定义类别名称和颜色
custom_class_names = ['< 1', '1--4', '> 4']  # 根据您实际类别替换
custom_colors = ['#3c9bc8', '#b8ddb1', '#fee59e']  # 蓝、橙、绿三色示例

plot_logreg_calibration_curve(
    model=best_stacking_models["LogisticRegression"],
    X_train=X_train_resampled,
    y_train=y_train_resampled,
    X_test=X_test_processed,
    y_test=y_test,
    classes=np.unique(y_train_resampled),
    class_names=custom_class_names,
    colors=custom_colors,
    n_bins=15
)


import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def compute_class_metrics(y_true, y_pred, y_proba, classes):
    metrics_list = []
    for idx, cls in enumerate(classes):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        y_proba_cls = y_proba[:, idx]

        accuracy = accuracy_score(y_true_bin, y_pred_bin)
        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        try:
            auc = roc_auc_score(y_true_bin, y_proba_cls)
        except ValueError:
            auc = 0.5

        metrics_list.append([accuracy, precision, recall, f1, auc])
    return np.array(metrics_list)

def compute_macro_metrics(metrics_mat):
    # metrics_mat shape: (n_classes, n_metrics)
    return np.mean(metrics_mat, axis=0)

def bootstrap_macro_metrics(y_true, y_pred, y_proba, classes, n_bootstrap=1000, random_state=42):
    np.random.seed(random_state)

    # 确保输入是 numpy 数组，避免 pandas 索引问题
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    n_samples = len(y_true)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    boot_metrics = []

    for i in range(n_bootstrap):
        indices = resample(range(n_samples), replace=True, n_samples=n_samples)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        y_proba_bs = y_proba[indices]

        cls_metrics = compute_class_metrics(y_true_bs, y_pred_bs, y_proba_bs, classes)
        macro_metrics = compute_macro_metrics(cls_metrics)
        boot_metrics.append(macro_metrics)

    boot_metrics = np.array(boot_metrics)  # shape: (n_bootstrap, n_metrics)
    ci_lower = np.percentile(boot_metrics, 2.5, axis=0)
    ci_upper = np.percentile(boot_metrics, 97.5, axis=0)
    macro_mean = np.mean(boot_metrics, axis=0)
    return macro_mean, ci_lower, ci_upper, metrics_names


# 主流程
unique_classes = np.unique(y_test)
y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# 计算原始宏平均指标（自代码中已有方法）
metrics_mat = compute_class_metrics(y_test, y_pred, y_proba, unique_classes)
macro_metrics_orig = compute_macro_metrics(metrics_mat)

# 计算bootstrap置信区间
macro_mean, ci_lower, ci_upper, metrics_names = bootstrap_macro_metrics(
    y_test, y_pred, y_proba, unique_classes, n_bootstrap=1000)

# 构建DataFrame保存结果
df_results = pd.DataFrame({
    'Metric': metrics_names,
    'Macro Mean': macro_mean,
    '95% CI Lower': ci_lower,
    '95% CI Upper': ci_upper
})

# 保存到Excel
output_excel_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\stacking-macro_metrics_with_CI.xlsx"
df_results.to_excel(output_excel_path, index=False)

print(f"宏平均指标及95%置信区间已保存到: {output_excel_path}")
print(df_results)


import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

# 假设已有：
# model = best_stacking_models["LogisticRegression"]
# X_test, y_test

unique_classes = np.unique(y_test)
n_classes = len(unique_classes)

# 多分类标签二值化，用于AUC计算
y_test_bin = label_binarize(y_test, classes=unique_classes)
y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']

# 为每个类别计算指标
metrics_list = []
for idx, cls in enumerate(unique_classes):
    # 对于单类别计算：
    # Accuracy：准确率计算整体分类正确率，针对单类别通常是所有样本中预测对和真实对该类别的综合考量
    # 这里用“该类别为正类”的二分类准确率计算（您也可根据需求调整）

    # 构造二分类标签
    y_true_bin = (y_test == cls).astype(int)
    y_pred_bin = (y_pred == cls).astype(int)
    y_proba_cls = y_proba[:, idx]

    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_true_bin, y_proba_cls)
    except ValueError:
        auc = 0.5  # 当单类别AUC无法计算时（通常是缺少正负样本），设置默认值

    metrics_list.append([accuracy, precision, recall, f1, auc])

metrics_mat = np.array(metrics_list)

# 类别标签和图例名称
# 4. 类别和图例名称处理
all_classes = df['Class'].unique()

custom_legend_dict = {
    0: '< 1',
    1: '1--4',
    2: '> 4',
    # 需要时可以继续补充
}
legend_names_all = [custom_legend_dict.get(cls, str(cls)) for cls in all_classes]

# 定义颜色字典
cmap_hex_dict = {
    "Stacking Model": ['#FDC3C8', '#F46E79', '#AA0A17']
}

model_name = "Stacking Model"

# 取颜色
if model_name in cmap_hex_dict:
    colors_this = cmap_hex_dict[model_name]
else:
    cmap = plt.cm.get_cmap('tab10', n_classes)
    colors_this = [cmap(i) for i in range(n_classes)]

# 确保颜色数量够用
if len(colors_this) < n_classes:
    times = n_classes // len(colors_this) + 1
    colors_this = (colors_this * times)[:n_classes]

output_path = r"C:\Users\ACER\Desktop\test\stt\picture\BM\radar\stack_radar.jpg"

# 调用plot_radar
plot_radar(
    metrics=metrics_mat,
    class_labels=all_classes,
    metric_labels=metrics_names,
    legend_names=legend_names_all,
    colors=colors_this,
    legend_loc='upper center',
    title_text=model_name,
    title_bg_width=0.8,
    title_bg_linewidth=2,
    title_bg_color='#AA0A17',
    title_text_color='#AA0A17',
    title_y=1.02,
    output_path= output_path
)


def plot_dca_for_meta_lr(y_true, y_pred_proba, unique_classes, titles=None):
    """
    仅绘制逻辑回归元学习器的多类别决策曲线分析（DCA）。
    
    参数：
    - y_true: 真实标签，shape (n_samples,)
    - y_pred_proba: 逻辑回归元学习器预测概率，shape (n_samples, n_classes)
    - unique_classes: 类别列表或数组
    - titles: 子图标题列表，长度应等于类别数，默认None
    """
    total_samples = len(y_true)
    n_classes = len(unique_classes)
    plt.figure(figsize=(5 * (n_classes + 1), 5))  # 多加一列用于宏平均曲线

    if titles is None:
        titles = [f'Class {cls}' for cls in unique_classes]
    elif len(titles) != n_classes:
        raise ValueError("参数 titles 长度应与 unique_classes 长度一致")

    thresholds = np.linspace(0.01, 0.99, 100)  # 阈值避免0和1

    macro_net_benefit = np.zeros_like(thresholds)

    for cls_idx, cls in enumerate(unique_classes):
        plt.subplot(1, n_classes + 1, cls_idx + 1)

        pos_total = np.sum(y_true == cls)
        net_benefit = []

        probs = y_pred_proba[:, cls_idx]

        for pt in thresholds:
            y_pred_threshold = (probs >= pt).astype(int)

            tp = np.sum((y_true == cls) & (y_pred_threshold == 1))
            fp = np.sum((y_true != cls) & (y_pred_threshold == 1))

            nb = (tp / total_samples) - (fp / total_samples) * (pt / (1 - pt))
            net_benefit.append(nb)

        net_benefit = np.array(net_benefit)
        macro_net_benefit += net_benefit

        plt.plot(thresholds, net_benefit, lw=2, label='Meta model',color='#AA0A17')

        prevalence = pos_total / total_samples if total_samples > 0 else 0
        treat_all_net_benefit = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        plt.plot(thresholds, treat_all_net_benefit, color='black', linestyle='--', lw=2, label='Treat All')

        plt.axhline(0, color='gray', linestyle='--', lw=2, label='Treat None')

        plt.title(titles[cls_idx],fontsize=20)
        plt.xlabel('Threshold Probability',fontsize=18)
        plt.ylabel('Net Benefit',fontsize=18)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.legend(fontsize=16)
        plt.ylim(bottom=-0.05, top=0.4)

    plt.subplot(1, n_classes + 1, n_classes + 1)

    avg_net_benefit = macro_net_benefit / n_classes
    plt.plot(thresholds, avg_net_benefit, lw=2, label='Meta model',color='#AA0A17')

    prevalence_all = np.mean([np.sum(y_true == cls) / total_samples for cls in unique_classes])
    treat_all_macro = prevalence_all - (1 - prevalence_all) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, treat_all_macro, color='black', linestyle='--', lw=2, label='Treat All')

    plt.axhline(0, color='gray', linestyle='--', lw=2, label='Treat None')

    plt.title('Macro Average DCA',fontsize=20)
    plt.xlabel('Threshold Probability',fontsize=18)
    plt.ylabel('Net Benefit',fontsize=18)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize=16)
    plt.ylim(bottom=-0.05, top=0.4)

    plt.tight_layout()
    #plt.savefig(r"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking-DCA.jpg", dpi=600, bbox_inches='tight')
    plt.show()
# 1. 获取逻辑回归元学习器模型
meta_lr_model = best_stacking_models["LogisticRegression"]

# 2. 预测测试集概率
y_pred_proba_meta_lr = meta_lr_model.predict_proba(X_test_processed)

# 3. 确定唯一类别列表（与训练时一致）
unique_classes = np.unique(y_test)

# 4. 定义类别对应的标题（可根据实际类别调整）
titles = ['DCA Curve(< 1)', 'DCA Curve(1--4)', 'DCA Curve(> 4)']

# 5. 调用绘图函数
plot_dca_for_meta_lr(y_test, y_pred_proba_meta_lr, unique_classes, titles=titles)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_logistic_regression_pr_curve(y_true, y_pred_proba, unique_classes, titles=None, colors=None, model_name='Logistic Regression'):
    """
    绘制逻辑回归多类别Precision-Recall曲线，支持自定义曲线颜色。

    参数：
    - y_true: array-like, shape (n_samples,) ，真实标签
    - y_pred_proba: ndarray, shape (n_samples, n_classes) ，逻辑回归预测概率
    - unique_classes: list/array，类别标签列表
    - titles: list，类别子图标题，长度应等于unique_classes长度
    - colors: list，颜色列表，长度应为 len(unique_classes)+1，最后一个为macro-average颜色
    - model_name: str，模型名称，用于图例显示
    """
    n_classes = len(unique_classes)
    plt.figure(figsize=(5 * (n_classes + 1), 5))

    if titles is None:
        titles = [f'Class {cls}' for cls in unique_classes]
    elif len(titles) != n_classes:
        raise ValueError("参数 titles 长度应与 unique_classes 长度一致")

    if colors is None:
        # 默认颜色，利用tab10色盘
        default_colors = plt.cm.get_cmap('tab10').colors
        if n_classes + 1 <= len(default_colors):
            colors = default_colors[:n_classes + 1]
        else:
            colors = [default_colors[i % len(default_colors)] for i in range(n_classes + 1)]
    else:
        if len(colors) != n_classes + 1:
            raise ValueError("参数 colors 长度应为类别数 + 1（宏平均）")

    # 每个类别PR曲线
    for cls_idx, cls in enumerate(unique_classes):
        plt.subplot(1, n_classes + 1, cls_idx + 1)

        y_true_binary = (y_true == cls).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, cls_idx])
        avg_precision = average_precision_score(y_true_binary, y_pred_proba[:, cls_idx])

        plt.plot(recall, precision, color=colors[cls_idx],
                 label=f'Meta Model (AP={avg_precision:.2f})', linewidth=2)
        plt.title(titles[cls_idx],fontsize=20)
        plt.xlabel('Recall',fontsize=18)
        plt.ylabel('Precision',fontsize=18)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.legend(loc='lower right',fontsize=16)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

    # 宏平均PR曲线绘制 (micro-average)
    plt.subplot(1, n_classes + 1, n_classes + 1)
    # 构造多标签二值矩阵
    y_true_bin = np.zeros((len(y_true), n_classes))
    for i, c in enumerate(unique_classes):
        y_true_bin[:, i] = (y_true == c).astype(int)

    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    avg_precision = average_precision_score(y_true_bin, y_pred_proba, average='micro')

    plt.plot(recall, precision, color=colors[-1],
             label=f'Meta Model (macro-avg AP={avg_precision:.2f})', linewidth=2)
    plt.title('Macro Average PR Curve',fontsize=20)
    plt.xlabel('Recall',fontsize=18)
    plt.ylabel('Precision',fontsize=18)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.legend(loc='lower right',fontsize=16)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.tight_layout()
    output_path = r"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking_PR_curve.jpg"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

# ---------------- 调用示例 ----------------

# 假设best_stacking_models里有训练好的逻辑回归元学习器，键名为'LogisticRegression'或类似
model_key = 'LogisticRegression'  # 请根据您字典中的key调整

y_pred_proba_lr = best_stacking_models[model_key].predict_proba(X_test_processed)
unique_classes = np.unique(y_test)

titles = ['Precision-Recall Curve(< 1)', 'Precision-Recall Curve(1--4)', 'Precision-Recall Curve(> 4)']
colors = ['#AA0A17','#AA0A17','#AA0A17','#AA0A17']  # 蓝，橙，绿，红 (最后红色为宏平均)

plot_logistic_regression_pr_curve(y_test, y_pred_proba_lr, unique_classes, titles=titles, colors=colors, model_name='Logistic Regression')


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_learner_names = ['lr', 'rf', 'adaboost']
custom_names = ['Logistic Regression', 'Random Forest', 'AdaBoost']

per_learner_feature_count = 3
# 自定义的类别标签，替换这里为您需要的字符串列表，长度必须等于per_learner_feature_count
custom_class_names = ['very high-risk', 'high-risk', 'low-risk']

np.random.seed(0)
coef = np.random.randn(len(base_learner_names) * per_learner_feature_count)

coef_2d = coef.reshape(len(base_learner_names), per_learner_feature_count)

importance_list = []
for i, learner_name in enumerate(base_learner_names):
    coef_slice = coef_2d[i, :]
    importance = np.sum(np.abs(coef_slice))
    importance_list.append((custom_names[i], importance))

feature_importances_df = pd.DataFrame(importance_list, columns=['Base Learner', 'Importance'])
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8), dpi=600)
plt.barh(feature_importances_df['Base Learner'], feature_importances_df['Importance'], color='steelblue')
plt.xlabel('Importance', fontsize=26)
plt.title('Feature Importance by Base Learners', fontsize=28)
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=24)
plt.gca().invert_yaxis()
plt.tight_layout()
output_dir = r"C:\Users\ACER\Desktop\test\stt\picture\BM"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "stacking-feature important.jpg"), dpi=600, bbox_inches='tight')
plt.show()

# 使用自定义类别标签作为行索引
coef_data = pd.DataFrame(index=custom_class_names)
for i, learner_name in enumerate(base_learner_names):
    coef_slice = coef_2d[i, :]
    coef_data[custom_names[i]] = coef_slice

plt.figure(figsize=(12, 8), dpi=600)
sns.heatmap(coef_data, annot=True, cmap='coolwarm', center=0, linewidths=0.5, annot_kws={"size": 20})
plt.title('Coefficients for Each Base Learner and Class Probability', fontsize=26)
plt.xlabel('Base Learner',fontsize=24)
plt.ylabel('ADT Resistance Risk',fontsize=24)
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)# 您想显示的y轴标签
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "stacking-hotmap.jpg"), dpi=600, bbox_inches='tight')
plt.show()


import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

save_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\combine\stacking_logistic_regression_model.pkl"
stacking_model = joblib.load(save_path)
print("逻辑回归元学习器已加载。")

feature_names = numeric_features + categorical_features

# 假设 stacking_model 是已经训练好的 sklearn StackingClassifier
# X_train_resampled, X_test_processed 是已准备好的 numpy 数组或类似结构
# feature_names 是特征名列表，长度对应特征数量

# 1. 缩小背景数据规模，避免KernelExplainer计算爆炸
background_size = 50
if X_train_resampled.shape[0] > background_size:
    np.random.seed(42)
    idx = np.random.choice(X_train_resampled.shape[0], background_size, replace=False)
    background_data = X_train_resampled[idx]
else:
    background_data = X_train_resampled

# 2. 选择用于解释的测试样本（示例5条）
X_to_explain = X_test_processed[:100]

def model_predict_proba(X):
    X = np.array(X)
    proba = stacking_model.predict_proba(X)
    return proba

# 3. 创建 KernelExplainer
explainer = shap.KernelExplainer(model_predict_proba, background_data)

print("开始计算 SHAP 值，可能耗时较长...")
shap_values = explainer.shap_values(X_to_explain)
# shap_values 是三维数组，形状 (样本数, 特征数, 类别数)

print(f"shap_values 类型: {type(shap_values)}")
print(f"shap_values shape: {np.array(shap_values).shape}")

# 4. 根据输出 shape 检查维度
# 注意：KernelExplainer 返回的 shap_values 对多分类是一个列表，长度等于类别数，每个元素是二维数组 (样本数, 特征数)
# 但有时会直接返回三维数组，根据版本不同有所差异
# 这里我们先转换为 np.array 方便处理：

shap_values = np.array(shap_values)  # 转为 numpy 数组，期待形状 (类别数, 样本数, 特征数)

print(f"转换为 numpy 数组后 shap_values shape: {shap_values.shape}")
# 例如 (3, 5, 27) 表示3个类别，5个样本，27个特征

class_idx = 0  # 选择类别0

shap_values_class0 = shap_values[:, :, class_idx]  # shape = (样本数, 特征数)

assert shap_values_class0.shape == X_to_explain.shape, "SHAP值与输入数据形状不匹配"

shap.summary_plot(shap_values_class0, X_to_explain, feature_names=feature_names)



import numpy as np
import matplotlib.pyplot as plt

# 假设 shap_values 形状为 (样本数, 特征数, 类别数)
# feature_names 是特征名列表，长度等于特征数
# 选择分析的类别索引，例如类别0
class_idx = 0

# 取类别 class_idx 下所有样本的 SHAP 值，形状 (样本数, 特征数)
shap_vals_for_class = shap_values[:, :, class_idx]

# 计算每个特征的平均绝对SHAP值，表示总体贡献度
mean_abs_shap = np.mean(np.abs(shap_vals_for_class), axis=0)  # 形状 (特征数,)

# 按贡献排序，得到排序索引
sorted_idx = np.argsort(mean_abs_shap)[::-1]

# 排序后的特征名和对应贡献值
sorted_feature_names = [feature_names[i] for i in sorted_idx]
sorted_mean_abs_shap = mean_abs_shap[sorted_idx]

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_mean_abs_shap, color='skyblue')
plt.xlabel('平均绝对SHAP值')
plt.title(f'feature important (class {class_idx})')
plt.gca().invert_yaxis()  # 反转y轴，让最大值在上方
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import re

feature_names = numeric_features + categorical_features

def clean_filename(filename):
    """
    清理文件名中的非法字符
    """
    cleaned_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    cleaned_filename = cleaned_filename.strip()
    return cleaned_filename if cleaned_filename else 'unnamed_file'


def plot_stacking_feature_importance(stacking_model, X_test_processed, y_test, feature_names, top_n=27):
    """
    计算 stacking 模型的 SHAP 值并绘制组合图（蜂巢点图+顶部条形图）
    
    :param stacking_model: 已训练好的 stacking 模型对象（需有 predict_proba 方法）
    :param X_test_processed: 测试集特征，numpy数组或等价格式
    :param y_test: 测试集标签，1维数组
    :param feature_names: 特征名列表，长度应与 X_test_processed 的列数一致
    :param top_n: 显示的最重要特征数量
    """

    X_df = pd.DataFrame(X_test_processed, columns=feature_names)
    unique_classes = np.unique(y_test)
    class_names = [str(c) for c in unique_classes]

    # 默认分析第0个类别
    class_idx = 2

    # 背景数据抽样，避免KernelExplainer计算过慢
    background_size = 100
    if X_df.shape[0] > background_size:
        np.random.seed(42)
        idx = np.random.choice(X_df.shape[0], background_size, replace=False)
        background_data = X_df.iloc[idx]
    else:
        background_data = X_df

    def model_predict_proba(X):
        X = np.array(X)
        return stacking_model.predict_proba(X)

    print("开始计算 stacking 模型的 SHAP 值，可能耗时较长...")
    explainer = shap.KernelExplainer(model_predict_proba, background_data)
    shap_values = explainer.shap_values(X_df)

    # shap_values 形状 (样本数, 特征数, 类别数)
    shap_vals_class = shap_values[:, :, class_idx]

    assert shap_vals_class.shape[0] == X_df.shape[0], "样本数不匹配"
    assert shap_vals_class.shape[1] == X_df.shape[1], "特征数不匹配"

    # 计算平均绝对 SHAP 值，选出 top_n 特征索引
    mean_abs_shap = np.abs(shap_vals_class).mean(axis=0)
    topn_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    # 直接用原始 feature_names，根据索引获取对应名称
    topn_feature_names = [feature_names[i] for i in topn_idx]
    X_topn = X_df.iloc[:, topn_idx]

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)

    shap.summary_plot(
        shap_vals_class[:, topn_idx],
        X_topn,
        feature_names=topn_feature_names,
        plot_type="dot",
        show=False,
        color_bar=True
    )
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])

    ax1 = plt.gca()
    ax2 = ax1.twiny()

    shap.summary_plot(
        shap_vals_class[:, topn_idx],
        X_topn,
        plot_type="bar",
        show=False
    )
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])

    ax2.axhline(y=top_n, color='gray', linestyle='-', linewidth=1)

    for bar in ax2.patches:
        bar.set_alpha(0.2)

    ax1.set_xlabel('Shapley Value (Impact on Model Output)', fontsize=18)
    ax2.set_xlabel(f'Feature Importance for Stacking Model', fontsize=18)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax1.set_ylabel('Features', fontsize=18)

    plt.tight_layout()

    output_path = rf"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking-StakingModel-combined-4.jpg"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"图像已保存到: {output_path}")
    plt.show()

plot_stacking_feature_importance(stacking_model, X_test_processed, y_test, feature_names, top_n=27)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def plot_stacking_shap_heatmap_first_sample(stacking_model, X_test_processed, feature_names, class_names=None,
                                            feature_start=0, feature_end=None,
                                            vmin=None, vmax=None,
                                            save_path=None):
    """
    绘制 stacking 模型的 SHAP 值热力图（取每个类别第一个样本的SHAP值，保留正负值，红蓝配色）

    参数：
    - stacking_model: 已训练的模型，需支持 predict_proba
    - X_test_processed: 测试集特征数组或DataFrame
    - feature_names: 特征名列表
    - class_names: 类别名列表，默认 ['<1','1--4','>4']
    - feature_start, feature_end: 只显示部分特征，索引范围 [start:end)
    - vmin, vmax: 颜色条的数值范围，不指定则自动根据数据范围设置
    - save_path: 保存图像路径，不指定则不保存
    """

    if class_names is None:
        class_names = ['<1', '1--4', '>4']

    X_df = pd.DataFrame(X_test_processed, columns=feature_names)

    background_size = 100
    if X_df.shape[0] > background_size:
        np.random.seed(42)
        idx = np.random.choice(X_df.shape[0], background_size, replace=False)
        background_data = X_df.iloc[idx]
    else:
        background_data = X_df

    def model_predict_proba(X):
        return stacking_model.predict_proba(np.array(X))

    print("开始计算 SHAP 值，可能耗时较长...")
    explainer = shap.KernelExplainer(model_predict_proba, background_data)
    shap_values = explainer.shap_values(X_df)
    print(f"shap_values shape: {shap_values.shape}")

    num_classes = shap_values.shape[2]
    assert num_classes == len(class_names), "类别数量与 class_names 长度不匹配"

    # 取每个类别第一个样本的SHAP值，直接组成DataFrame
    first_sample_shap_list = []
    for c_idx in range(num_classes):
        # 取第一个样本的shap值 (shape: 特征数,)
        first_sample_shap = shap_values[2, :, c_idx]
        print(f"类别 {class_names[c_idx]} 第一个样本 SHAP 形状: {first_sample_shap.shape}")
        first_sample_shap_list.append(first_sample_shap)

    heatmap_df = pd.DataFrame(
        data=np.array(first_sample_shap_list).T,
        index=feature_names,
        columns=class_names
    )

    # 选取部分特征显示
    if feature_end is None or feature_end > len(feature_names):
        feature_end = len(feature_names)
    heatmap_df_subset = heatmap_df.iloc[feature_start:feature_end, :]

    if vmin is None or vmax is None:
        abs_max = np.abs(heatmap_df_subset.values).max()
        vmin = -abs_max
        vmax = abs_max

    plt.figure(figsize=(12, max(8, 0.2 * heatmap_df_subset.shape[0])), dpi=120)
    sns.heatmap(
        heatmap_df_subset,
        cmap='bwr',
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'SHAP value (first sample)'},
        linewidths=0.5,
        linecolor='lightgray',
        annot_kws={"size": 24}
    )
    plt.title('Heatmap of SHAP Values for First Sample by Class', fontsize=20)
    plt.xlabel('ADT Resistance Risk',fontsize=20)
    plt.ylabel('Feature',fontsize=20)
    plt.tick_params(axis='x', labelsize=18)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {save_path}")

    plt.show()


# 调用示例（请确保变量名和您的环境一致）
plot_stacking_shap_heatmap_first_sample(
    stacking_model=model,
    X_test_processed=X_test,
    feature_names=feature_names,
    class_names=['very high-risk', 'high-risk', 'low-risk'],
    feature_start=0,
    feature_end=27,
    save_path=r"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking_shap_first_sample_heatmap.jpg"
)


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 假设 stacking_model 是训练好的 StackingClassifier 三分类模型
# X_test_processed, y_test 已准备好，feature_names 已定义

X_df = pd.DataFrame(X_test_processed, columns=feature_names)

unique_classes = np.unique(y_test)
class_names = [str(c) for c in unique_classes]

class_idx = 0  # 选择类别索引
sample_idx = 2  # 选择样本索引

print(f"\n--- SHAP 分析: Stacking 模型 ---")

# 这里使用 KernelExplainer 对 Stacking 模型的 predict_proba 进行解释
# 取背景样本，数量根据实际情况调整，越多越准确但越慢
background = shap.sample(X_df, 100)

explainer = shap.KernelExplainer(stacking_model.predict_proba, background)
shap_values = explainer.shap_values(X_df)

# shap_values 是 list，长度为类别数，形状均为 (num_samples, num_features)
# 转换为形状 (num_samples, num_features, num_classes)
shap_values = shap_values.transpose(0, 1, 2)

base_values = explainer.expected_value
if isinstance(base_values, (list, np.ndarray)) and len(base_values) > 1:
    base_val = base_values[class_idx]
else:
    base_val = base_values

assert shap_values.shape[0] == X_df.shape[0], "样本数不匹配"
assert shap_values.shape[1] == X_df.shape[1], "特征数不匹配"
assert shap_values.shape[2] == len(class_names), "类别数不匹配"

# 计算前10重要特征索引
mean_abs_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
top10_idx = np.argsort(mean_abs_shap)[::-1][:10]

top10_feature_names = [feature_names[i] for i in top10_idx]
X_top10 = X_df.iloc[:, top10_idx]

# 构造 Explanation 对象，用于 summary/beeswarm 图
shap_exp = shap.Explanation(
    values=shap_values[:, top10_idx, class_idx],
    base_values=base_val,
    data=X_top10,
    feature_names=top10_feature_names
)

# 绘制 summary/beeswarm 图（前10特征）
print(f"Stacking 模型 类别 '{class_names[class_idx]}' 的前10特征 beeswarm plot")
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_exp, show=True)
plt.savefig(r"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking-beeswarm-0.jpg", dpi=600, bbox_inches='tight')
plt.show()

# 选定样本的 SHAP 值和特征值（前10特征）
shap_vals_sample_top10 = shap_values[sample_idx, top10_idx, class_idx]
feature_sample_top10 = X_top10.iloc[sample_idx]

# 绘制 force plot
print(f"Stacking 模型 类别 '{class_names[class_idx]}' 样本 {sample_idx} 的 force_plot（前10特征）")
plt.figure(figsize=(12, 3))
shap.plots.force(base_val, shap_vals_sample_top10, feature_sample_top10, matplotlib=True)
plt.tight_layout()
plt.savefig(r"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking-force-0.jpg", dpi=600, bbox_inches='tight')
plt.show()

# 绘制 waterfall plot
print(f"Stacking 模型 类别 '{class_names[class_idx]}' 样本 {sample_idx} 的 waterfall_plot（前10特征）")
plt.figure(figsize=(12, 8))
shap.plots._waterfall.waterfall_legacy(base_val, shap_vals_sample_top10, feature_names=top10_feature_names)
plt.tight_layout()
plt.savefig(r"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking-waterfall-0.jpg", dpi=600, bbox_inches='tight')
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import re

# 组装特征名列表（顺序务必与训练数据顺序保持一致）
feature_names = numeric_features + categorical_features

def clean_filename(filename):
    """清理文件名中的特殊字符"""
    cleaned_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    cleaned_filename = cleaned_filename.strip()
    return cleaned_filename if cleaned_filename else 'unnamed_file'
    
# 绘制基学习器特征重要性的组合图
def plot_combined_feature_importance(best_models, X_test_processed, y_test, top_n=10):
    """
    绘制基学习器特征重要性的组合图（条形图 + 蜂窝图）
    :param best_models: 包含训练好的基学习器模型的字典
    :param X_test_processed: 测试集特征数组
    :param y_test: 测试集标签
    :param top_n: 显示最重要的前N个特征
    """
    # 将测试集特征转换为DataFrame
    X_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # 获取类别名称
    unique_classes = np.unique(y_test)
    class_names = [str(c) for c in unique_classes]
    
    # 选择类别索引和样本索引
    class_idx = 2  # 选择类别索引
    sample_idx = 2  # 选择样本索引

    # 遍历每个模型
    for model_name, model in best_models.items():
        print(f"\n--- 特征重要性分析: {model_name} ---")

        # 创建主图（用来画蜂巢图）
        fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)

        # 计算SHAP值
        if model_name == "Logistic Regression":
            masker = shap.maskers.Independent(data=X_df)
            explainer = shap.LinearExplainer(model, masker=masker)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).transpose(1, 2, 0)
        elif model_name == "Random Forest":
            background = shap.sample(X_df, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).transpose(1, 2, 0)
        elif model_name == "AdaBoost":
            explainer = shap.KernelExplainer(model.predict_proba, X_df)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).transpose(1, 2, 0)
        else:
            print(f"模型 {model_name} 不支持 SHAP 解释。")
            continue

        base_values = explainer.expected_value
        if isinstance(base_values, (list, np.ndarray)) and len(base_values) > 1:
            base_val = base_values[class_idx]
        else:
            base_val = base_values

        # 断言维度正确
        assert shap_values.shape[0] == X_df.shape[0], "样本数不匹配"
        assert shap_values.shape[1] == X_df.shape[1], "特征数不匹配"
        assert shap_values.shape[2] == len(class_names), "类别数不匹配"

        # 计算前10重要特征索引
        mean_abs_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
        top10_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

        top10_feature_names = [feature_names[i] for i in top10_idx]
        X_top10 = X_df.iloc[:, top10_idx]

        # 在主图上绘制蜂巢图，并保留热度条
        shap.summary_plot(shap_values[:, top10_idx, class_idx], X_top10, feature_names=top10_feature_names, plot_type="dot", show=False, color_bar=True)
        plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，留出右侧空间放热度条

        # 获取共享的 y 轴
        ax1 = plt.gca()

        # 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
        ax2 = ax1.twiny()
        shap.summary_plot(shap_values[:, top10_idx, class_idx], X_top10, plot_type="bar", show=False)
        plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐

        # 在顶部 X 轴添加一条横线
        ax2.axhline(y=top_n, color='gray', linestyle='-', linewidth=1)  # 注意y值应该对应顶部

        # 调整透明度
        bars = ax2.patches  # 获取所有的柱状图对象
        for bar in bars:
            bar.set_alpha(0.2)  # 设置透明度

        # 设置两个x轴的标签
        ax1.set_xlabel('Shapley Value (Impact on Model Output)', fontsize=16)
        ax2.set_xlabel(f'Top {top_n} Feature Importance for {model_name}', fontsize=16)

        # 移动顶部的 X 轴，避免与底部 X 轴重叠
        ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
        ax2.xaxis.tick_top()  # 将刻度也移动到顶部

        # 设置y轴标签
        ax1.set_ylabel('Features', fontsize=16)

        # 调整布局
        plt.tight_layout()

        # 保存图形
        output_path = rf"C:\Users\ACER\Desktop\test\stt\picture\BM\stacking-{clean_filename(model_name)}-1.jpg"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

# 执行绘图
plot_combined_feature_importance(best_models, X_test_processed, y_test)



