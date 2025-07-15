import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
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
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import ADASYN
from catboost import CatBoostClassifier

plt.rcParams['font.family'] = 'Times New Roman'
random_seed = 42
np.random.seed(random_seed)

dataFile = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\radiology\ML\feature-final-selected-lasso-frecv.xlsx"
data = pd.read_excel(dataFile)
X = data.iloc[:, :-1]
y = data['label']
unique_classes = np.unique(y)
n_classes = len(unique_classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed, stratify=y
)
adasyn = ADASYN(random_state=random_seed)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)
models = {
    "Logistic Regression": LogisticRegression(multi_class='ovr', max_iter=1000, random_state=random_seed),
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
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    },
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance']
    },
    "Naive Bayes": {},
    "AdaBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "CatBoost": {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [3, 5, 7]
    }
}


def bootstrap_ci(y_true, y_pred_label=None, y_pred_prob=None, metric_func=None, 
                 n_bootstraps=1000, alpha=0.05, multi_class=False, **metric_kwargs):
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
        if multi_class:
            y_true_bin = label_binarize(y_true[indices], classes=unique_classes)
            score = metric_func(y_true_bin, y_pred_prob[indices], **metric_kwargs)
        else:
            score = metric_func(y_true[indices], y_pred_label[indices], **metric_kwargs)
        scores.append(score)
    scores = np.array(scores)
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(scores)

    return mean_score, lower, upper
best_models = {}
results = []

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, 
                               scoring='roc_auc_ovr', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_resampled)
    
    best_models[model_name] = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    results.append({
        "Model": model_name,
        "Best Parameters": best_params,
        "Best Cross-Validated Macro AUC": best_score
    })

results_df = pd.DataFrame(results)
print("模型调优结果：")
print(results_df)
results_df.to_excel(r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\radiology\ML\model_tuning_results.xlsx", index=False)
performance_results = []
class_performance_results = []
y_test_bin = label_binarize(y_test, classes=unique_classes)

def bootstrap_ci_binary(y_true, y_pred_label=None, y_pred_prob=None, metric_func=None,
                        n_bootstraps=1000, alpha=0.05, **metric_kwargs):
    
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
            yp_sample = y_pred_prob[indices]
        else:
            yp_sample = y_pred_label[indices] if y_pred_label is not None else y_pred_prob[indices]
        try:
            score = metric_func(yt_sample, yp_sample, **metric_kwargs)
        except ValueError:
            continue
        scores.append(score)

    scores = np.array(scores)
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(scores)

    return mean_score, lower, upper

print("开始计算各模型性能及置信区间...")
def predict_with_threshold(proba, threshold=0.5):
    candidates = np.where(proba > threshold)[1] if proba.ndim == 2 else (proba > threshold)
    y_pred = []
    for p in proba:
        above_thresh = np.where(p > threshold)[0]
        if len(above_thresh) == 0:
            y_pred.append(np.argmax(p))
        elif len(above_thresh) == 1:
            y_pred.append(above_thresh[0])
        else:
            best_idx = above_thresh[np.argmax(p[above_thresh])]
            y_pred.append(best_idx)
    return np.array(y_pred)
    
for model_name, model in best_models.items():
    print(f"评估 {model_name} ...")
    y_pred_proba = model.predict_proba(X_test_scaled)
    y_pred = predict_with_threshold(y_pred_proba, threshold=0.5)

    macro_f1_mean, macro_f1_low, macro_f1_high = bootstrap_ci(
        y_test, y_pred_label=y_pred, y_pred_prob=y_pred_proba, metric_func=f1_score,
        n_bootstraps=1000, alpha=0.05, multi_class=False, average='macro'
    )
    accuracy_mean, accuracy_low, accuracy_high = bootstrap_ci(
        y_test, y_pred_label=y_pred, metric_func=accuracy_score,
        n_bootstraps=1000, alpha=0.05, multi_class=False
    )
    precision_mean, precision_low, precision_high = bootstrap_ci(
        y_test, y_pred_label=y_pred, metric_func=precision_score,
        n_bootstraps=1000, alpha=0.05, multi_class=False, average='macro'
    )
    recall_mean, recall_low, recall_high = bootstrap_ci(
        y_test, y_pred_label=y_pred, metric_func=recall_score,
        n_bootstraps=1000, alpha=0.05, multi_class=False, average='macro'
    )
    roc_auc_mean, roc_auc_low, roc_auc_high = bootstrap_ci(
        y_test, y_pred_label=y_pred, y_pred_prob=y_pred_proba,
        metric_func=lambda yt, yp: roc_auc_score(yt, yp, multi_class='ovr', average='macro'),
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

performance_df.to_excel(r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\radiology\ML\R-model_performance_SVI_CI.xlsx", index=False)
class_performance_df.to_excel(r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\radiology\ML\class-performance-SVI-CI.xlsx", index=False)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f'Confusion Matrix: {model_name}')
plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

n_bootstraps = 1000
rng_seed = 42  
rng = np.random.RandomState(rng_seed)

plt.figure(figsize=(12, 8))

colors = ['#297fb8', '#f0c514', '#23a54f', '#f373e4', '#6e3383', '#cd6155', '#43735d', '#f97506']

for (model_name, model), color in zip(best_models.items(), colors):
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_roc_auc = auc(all_fpr, mean_tpr)
    
   
    bootstrapped_scores = []
    n_samples = y_test_bin.shape[0]
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_test_bin[indices], axis=0)) < n_classes:
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
plt.title('Macro-ROC Curves for All Models',fontsize=20)
plt.legend(loc="lower right",fontsize=14)
plt.grid(False)

output_path = r"C:\Users\ACER\Desktop\test\stt\picture\ML\ML Model ROC.jpg"
plt.savefig(output_path, dpi=600, format='jpg', bbox_inches='tight')
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

output_cm_dir = r"C:\Users\ACER\Desktop\test\stt\picture\ML\confusion_matrices"
os.makedirs(output_cm_dir, exist_ok=True)

unique_classes = np.unique(y)
assert len(unique_classes) == len(category_labels), "unique_classes和category_labels长度不匹配！"

for model_name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_labels)
    
    plt.figure(figsize=(12, 8))
    
    hex_colors = cmap_hex_dict.get(model_name, ['#e0e0e0', '#000000'])
    custom_cmap = LinearSegmentedColormap.from_list(f'{model_name}_cmap', hex_colors)

    ax = plt.gca()
    disp.plot(cmap=custom_cmap, values_format='d', ax=ax)
    for txt in ax.texts:
        txt.set_visible(False)
    
    im = ax.images[0]  

    plt.title(f"{model_name}", fontsize=28, fontweight='bold')
    plt.xticks()
    
    col_sum = cm.sum(axis=1, keepdims=True) 
    col_sum_safe = np.where(col_sum == 0, 1, col_sum)  
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
file_path = r"C:\Users\ACER\Desktop\test\stt\MODEL_DATE\radiology\ML\class-performance-SVI.xlsx"
df = pd.read_excel(file_path)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
def plot_radar(metrics, class_labels, metric_labels, legend_names=None, colors=None,
               legend_loc='upper center', title_text="Clinical Model",
               title_bg_width=1.0, title_bg_linewidth=1.5,
               title_bg_color='gray', title_text_color=None,
               title_y=1.0, fontsize=18, output_path=None):

    if title_text_color is None:
        title_text_color = title_bg_color 

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

    for idx, row in enumerate(metrics):
        vals = row[:sides].tolist()
        vals += vals[:1]  # 闭合
        angles_plot = np.append(pentagon_angles, pentagon_angles[0])
        ax.plot(angles_plot, vals, color=colors[idx], linewidth=2, label=str(legend_names[idx]))
        ax.scatter(angles_plot, vals, color=colors[idx], s=50, zorder=10, linewidth=0.5)
        ax.set_theta_offset(0)
        ax.set_theta_direction(1)
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_frame_on(False)
        ax.patch.set_visible(False)
        ax.set_rlim(0, 1)
        ax.set_yticklabels([])
        ax.set_xticks([])

    grid_radii = np.linspace(0.2, 1.0, 5)
    for r in grid_radii:
        xs = r * np.cos(pentagon_angles)
        ys = r * np.sin(pentagon_angles)
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        ax.plot(np.arctan2(ys, xs), np.hypot(xs, ys), color='gray', linestyle='dotted', linewidth=1, zorder=0)

   
    xs = np.cos(pentagon_angles)
    ys = np.sin(pentagon_angles)
    xs = np.append(xs, xs[0])
    ys = np.append(ys, ys[0])
    ax.plot(np.arctan2(ys, xs), np.hypot(xs, ys), color='gray', linestyle='-', linewidth=1.5, zorder=5)
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

   
    for angle in pentagon_angles:
        ax.plot([angle, angle], [0, 1], color='gray', linestyle='dotted', linewidth=1, zorder=0)

    label_r = 1.2
    for angle, label in zip(pentagon_angles, display_labels):
        ax.text(angle, label_r, label, fontsize=26,
                horizontalalignment='center',
                verticalalignment='center')
    legend = ax.legend(
        loc=legend_loc,
        bbox_to_anchor=(0.5, 1.27),
        ncol=len(class_labels),
        fontsize=24,
        frameon=False
    )
    for text, color in zip(legend.get_texts(), colors):
        text.set_color(color)
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

all_classes = df['Class'].unique()

custom_legend_dict = {
    0: 'very high-risk',
    1: 'high-risk',
    2: 'low-risk',
}
legend_names_all = [custom_legend_dict.get(cls, str(cls)) for cls in all_classes]
base_colors = ['#e74c3c', '#3498db', '#f39c12']
if len(all_classes) > len(base_colors):
    cmap = plt.cm.get_cmap('tab10', len(all_classes))
    colors_all_default = [cmap(i) for i in range(len(all_classes))]
else:
    colors_all_default = base_colors[:len(all_classes)]
model_colors_dict = {
    'Logistic Regression': ['#c5dcec', '#76aed1', '#297fb8'],
    'SVM': ['#f8e598', '#f5d75b', '#f0c514'],
    'Random Forest': ['#abe3be', '#6dc68e', '#23a54f'],
    'K-Nearest Neighbors': ['#f6b9e3','#f59be3','#f373e4'],
    'Naive Bayes':['#ccb0d9','#9d71ae','#6e3383'],
    'AdaBoost':['#f3cac6','#e39c96','#cd6155'],
    'Gradient Boosting':['#afdbde','#76a499','#43735d'],
    'CatBoost':['#f3c69f','#f79d51','#f97506']
}

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


output_dir = r"C:\Users\ACER\Desktop\test\stt\picture\ML\radar"
os.makedirs(output_dir, exist_ok=True)

models = df['Model'].unique()
for model in models:
    df_model = df[df['Model'] == model]
    df_model = df_model.set_index('Class').reindex(all_classes)
    metrics_mat = df_model[metrics_names].to_numpy()

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
        title_bg_color=title_color,    
        title_text_color=title_color,   
        title_y=1.01,
        output_path=output_file
    )




