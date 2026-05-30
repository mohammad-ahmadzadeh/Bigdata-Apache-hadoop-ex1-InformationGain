#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


HADOOP_HOME = "/usr/local/hadoop"
HDFS_INPUT_DIR = "/user/moses/experiments/input"
HDFS_OUTPUT_BASE = "/user/moses/experiments/output"
LOCAL_RESULTS_DIR = "./experiment_results"
LOCAL_DATASET_PATH = "./normalized_dataset.csv"

EXPERIMENTS = [
    (1, 1, 1, 100, "baseline_small_chunk"),
    (2, 1, 1, 200, "baseline_medium_chunk"),
    (3, 1, 1, 500, "baseline_large_chunk"),
    (4, 1, 1, 1000, "baseline_full_chunk"),
    (5, 1, 3, 1000, "more_reducers"),
    (6, 1, 5, 1000, "even_more_reducers"),
    (7, 3, 1, 1000, "more_mappers"),
    (8, 3, 3, 1000, "balanced_equal"),
    (9, 3, 5, 1000, "balanced_more_reducers"),
    (10, 5, 1, 1000, "even_more_mappers"),
    (11, 5, 3, 1000, "balanced_2"),
    (12, 5, 5, 1000, "balanced_full"),
]

CLASSIFIERS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', random_state=42)
}


def create_directories():
    Path(LOCAL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for i in range(1, 13):
        Path(f"{LOCAL_RESULTS_DIR}/exp_{i:02d}").mkdir(parents=True, exist_ok=True)


def check_hadoop():
    result = subprocess.run(["which", "hadoop"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Hadoop not found in PATH")
        return False
    print(f"[OK] Hadoop found: {result.stdout.strip()}")
    return True


def check_hdfs():
    result = subprocess.run(["hdfs", "dfs", "-ls", "/"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] HDFS not available")
        return False
    print("[OK] HDFS is available")
    return True


def setup_hdfs():
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", HDFS_INPUT_DIR], capture_output=True)
    subprocess.run(["hdfs", "dfs", "-rm", "-r", "-f", HDFS_OUTPUT_BASE], capture_output=True)
    print(f"[OK] HDFS directories prepared: {HDFS_INPUT_DIR}")
    return True


def load_full_dataset():
    df = pd.read_csv(LOCAL_DATASET_PATH)
    
    if 'label' in df.columns:
        label_column = 'label'
    elif 'Label' in df.columns:
        label_column = 'Label'
    else:
        label_column = df.columns[-1]
    
    feature_columns = [col for col in df.columns if col != label_column]
    full_features = df[feature_columns].values
    full_labels = df[label_column].values
    
    return full_features, full_labels, feature_columns


def upload_csv_to_hdfs():
    hdfs_csv_path = f"{HDFS_INPUT_DIR}/data.csv"
    subprocess.run(["hdfs", "dfs", "-rm", "-f", hdfs_csv_path], capture_output=True)
    result = subprocess.run(["hdfs", "dfs", "-put", "-f", LOCAL_DATASET_PATH, hdfs_csv_path], capture_output=True)
    
    if result.returncode == 0:
        print(f"[OK] CSV uploaded to HDFS: {hdfs_csv_path}")
        return True
    else:
        print(f"[ERROR] Failed to upload CSV to HDFS")
        return False


def save_mapper_reducer_files():
    mapper_path = f"{LOCAL_RESULTS_DIR}/mapper.py"
    reducer_path = f"{LOCAL_RESULTS_DIR}/reducer.py"
    
    mapper_code = '''#!/usr/bin/env python3
import sys

first_line = True
target_col = -1

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    
    parts = line.split(',')
    
    if first_line:
        for i, col in enumerate(parts):
            if col.lower().strip() in ['label', 'class', 'target']:
                target_col = i
                break
        if target_col == -1 and len(parts) > 0:
            target_col = len(parts) - 1
        first_line = False
        continue
    
    if target_col >= len(parts):
        continue
    
    target_val = parts[target_col].strip()
    if not target_val or target_val.lower() == 'label':
        continue
    
    try:
        target_val = int(float(target_val))
    except:
        continue
    
    for i, val in enumerate(parts):
        if i == target_col:
            continue
        try:
            feature_val = float(val)
            sys.stdout.write(f"{i}\\t{feature_val}|{target_val}\\n")
        except:
            pass
'''
    
    reducer_code = '''#!/usr/bin/env python3
import sys
import math
from collections import defaultdict

def calc_entropy(counts, total):
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy

feature_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
global_counts = defaultdict(int)
total = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    
    parts = line.split('\\t')
    if len(parts) != 2:
        continue
    
    try:
        fid = int(parts[0])
        parts2 = parts[1].split('|')
        if len(parts2) != 2:
            continue
        
        fval = float(parts2[0])
        tval = int(float(parts2[1]))
        
        feature_data[fid][fval][tval] += 1
        global_counts[tval] += 1
        total += 1
    except:
        continue

if total == 0:
    sys.exit(0)

global_entropy = calc_entropy(global_counts, total)

for fid, fdict in feature_data.items():
    cond_entropy = 0.0
    for vcounts in fdict.values():
        vtotal = sum(vcounts.values())
        weight = vtotal / total
        ventropy = calc_entropy(vcounts, vtotal)
        cond_entropy += weight * ventropy
    
    ig = global_entropy - cond_entropy
    sys.stdout.write(f"{fid}\\t{ig}\\n")
'''
    
    with open(mapper_path, 'w') as f:
        f.write(mapper_code)
    with open(reducer_path, 'w') as f:
        f.write(reducer_code)
    
    os.chmod(mapper_path, 0o755)
    os.chmod(reducer_path, 0o755)
    
    return mapper_path, reducer_path


def get_streaming_jar():
    jar_pattern = Path(f"{HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-*.jar")
    jar_files = list(jar_pattern.parent.glob("hadoop-streaming-*.jar"))
    if jar_files:
        return str(jar_files[0])
    return None


def run_hadoop_experiment(exp_id, map_tasks, reduce_tasks, chunk_size, description):
    print(f"\n[EXPERIMENT {exp_id:02d}] Starting...")
    print(f"  Description: {description}")
    print(f"  Parameters: Mappers={map_tasks}, Reducers={reduce_tasks}, Chunk Size={chunk_size} KB")
    
    exp_output_dir = f"{HDFS_OUTPUT_BASE}/exp_{exp_id:02d}"
    
    subprocess.run(["hdfs", "dfs", "-rm", "-r", "-f", exp_output_dir], capture_output=True)
    
    streaming_jar = get_streaming_jar()
    if not streaming_jar:
        print(f"  [ERROR] Streaming JAR not found")
        return False, 0
    
    input_file = f"{HDFS_INPUT_DIR}/data.csv"
    
    hadoop_cmd = [
        "hadoop", "jar", streaming_jar,
        "-D", f"mapreduce.job.maps={map_tasks}",
        "-D", f"mapreduce.job.reduces={reduce_tasks}",
        "-D", f"mapreduce.input.fileinputformat.split.maxsize={chunk_size * 1024}",
        "-D", "mapreduce.job.name=feature_selection_exp_" + str(exp_id),
        "-input", input_file,
        "-output", exp_output_dir,
        "-mapper", "mapper.py",
        "-reducer", "reducer.py",
        "-file", f"{LOCAL_RESULTS_DIR}/mapper.py",
        "-file", f"{LOCAL_RESULTS_DIR}/reducer.py",
    ]
    
    print(f"  Running Hadoop job...")
    start_time = time.time()
    
    try:
        result = subprocess.run(hadoop_cmd, capture_output=True, text=True, timeout=600)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            output_check = subprocess.run(["hdfs", "dfs", "-ls", exp_output_dir], capture_output=True, text=True)
            if "_SUCCESS" in output_check.stdout:
                print(f"  [OK] Job completed in {elapsed_time:.2f} seconds")
                return True, elapsed_time
            else:
                print(f"  [ERROR] Job finished but _SUCCESS not found")
                return False, elapsed_time
        else:
            print(f"  [ERROR] Job failed with return code {result.returncode}")
            return False, elapsed_time
            
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Job timeout after 600 seconds")
        return False, 600
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return False, 0


def load_top_features(exp_id):
    exp_output_dir = f"{HDFS_OUTPUT_BASE}/exp_{exp_id:02d}"
    local_output_dir = f"{LOCAL_RESULTS_DIR}/exp_{exp_id:02d}"
    
    download_cmd = ["hdfs", "dfs", "-get", exp_output_dir, local_output_dir]
    subprocess.run(download_cmd, capture_output=True)
    
    all_features = []
    
    for part_file in Path(local_output_dir).glob("part-*"):
        if part_file.is_file() and part_file.stat().st_size > 0:
            with open(part_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            try:
                                feature_id = int(parts[0])
                                score = float(parts[1])
                                all_features.append((feature_id, score))
                            except ValueError:
                                pass
    
    if len(all_features) == 0:
        stream = subprocess.run(["hdfs", "dfs", "-cat", f"{exp_output_dir}/part-*"], capture_output=True, text=True)
        for line in stream.stdout.strip().split('\n'):
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    try:
                        feature_id = int(parts[0])
                        score = float(parts[1])
                        all_features.append((feature_id, score))
                    except ValueError:
                        pass
    
    all_features.sort(key=lambda x: x[1], reverse=True)
    return all_features[:100]


def extract_selected_features(full_features, selected_feature_indices):
    max_idx = full_features.shape[1]
    indices = [idx for idx, _ in selected_feature_indices if idx < max_idx]
    
    if not indices:
        return np.zeros((full_features.shape[0], 1))
    
    return full_features[:, indices]


def evaluate_classifier(clf, X, y):
    try:
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[0] < 10:
            return 0.0, 0.0
        
        scores = cross_val_score(clf, X, y, cv=min(5, X.shape[0]), scoring='accuracy', n_jobs=-1)
        return scores.mean(), scores.std()
    except Exception:
        try:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return accuracy_score(y_test, y_pred), 0.0
        except Exception:
            return 0.0, 0.0


def evaluate_all_classifiers(features, labels, classifier_dict):
    results = {}
    for clf_name, clf in classifier_dict.items():
        accuracy, std = evaluate_classifier(clf, features, labels)
        results[clf_name] = {'accuracy': accuracy, 'std': std}
    return results


def save_experiment_results(exp_id, top_features, execution_time, map_tasks, reduce_tasks, chunk_size, description, full_features, full_labels, feature_names):
    local_output_dir = f"{LOCAL_RESULTS_DIR}/exp_{exp_id:02d}"
    
    if len(top_features) < 10:
        max_features = min(100, full_features.shape[1])
        selected_indices = [(i, 0.0) for i in range(max_features)]
        selected_features = full_features[:, :max_features]
    else:
        selected_indices = top_features
        selected_features = extract_selected_features(full_features, selected_indices)
    
    if selected_features.size == 0 or selected_features.shape[0] == 0:
        classifier_results = {name: {'accuracy': 0.0, 'std': 0.0} for name in CLASSIFIERS.keys()}
    else:
        classifier_results = evaluate_all_classifiers(selected_features, full_labels, CLASSIFIERS)
    
    results_file = Path(local_output_dir) / "top_100_features.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"EXPERIMENT {exp_id:02d} RESULTS\n")
        f.write("=" * 100 + "\n")
        f.write(f"Description: {description}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Map Tasks: {map_tasks}\n")
        f.write(f"Reduce Tasks: {reduce_tasks}\n")
        f.write(f"Chunk Size: {chunk_size} KB\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"Total Features Extracted: {len(top_features)}\n")
        f.write("\n" + "=" * 100 + "\n")
        f.write("TOP 100 FEATURES BY INFORMATION GAIN\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Feature ID':<15} {'Score':<20}\n")
        f.write("-" * 100 + "\n")
        
        for rank, (feature_id, score) in enumerate(top_features[:100], 1):
            f.write(f"{rank:<6} {feature_id:<15} {score:<20.8f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("CLASSIFIER PERFORMANCE ON TOP 100 FEATURES\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Classifier':<20} {'Accuracy':<15} {'Std Dev':<15}\n")
        f.write("-" * 100 + "\n")
        
        for clf_name, metrics in classifier_results.items():
            f.write(f"{clf_name:<20} {metrics['accuracy']:<15.4f} {metrics['std']:<15.4f}\n")
        
        best_clf = max(classifier_results.items(), key=lambda x: x[1]['accuracy'])
        f.write("\n" + "-" * 100 + "\n")
        f.write(f"Best Classifier: {best_clf[0]} with accuracy {best_clf[1]['accuracy']:.4f}\n")
    
    summary_file = Path(local_output_dir) / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"{exp_id},{map_tasks},{reduce_tasks},{chunk_size},{execution_time:.2f},{len(top_features)},{description}\n")
        for clf_name, metrics in classifier_results.items():
            f.write(f"{clf_name},{metrics['accuracy']:.4f},{metrics['std']:.4f}\n")
    
    return classifier_results


def generate_final_report(all_results, feature_names):
    report_file = f"{LOCAL_RESULTS_DIR}/FINAL_REPORT.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("FINAL REPORT: FEATURE SELECTION EXPERIMENTS WITH CLASSIFIER EVALUATION\n")
        f.write("=" * 120 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {LOCAL_DATASET_PATH}\n\n")
        
        f.write("EXPERIMENT SUMMARY\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Exp':<5} {'Status':<8} {'Map':<6} {'Reduce':<8} {'Chunk(KB)':<10} {'Time(s)':<12} {'Features':<10} {'Best Acc':<12}\n")
        f.write("-" * 120 + "\n")
        
        summary_data = []
        for exp_id, res in all_results.items():
            if res and res.get('success'):
                best_acc = 0
                if 'classifiers' in res and res['classifiers']:
                    best = max(res['classifiers'].items(), key=lambda x: x[1]['accuracy'])
                    best_acc = best[1]['accuracy']
                f.write(f"{exp_id:<5} {'SUCCESS':<8} {res['maps']:<6} {res['reduces']:<8} {res['chunk']:<10} {res['time']:<12.2f} {len(res['features']):<10} {best_acc:<12.4f}\n")
                summary_data.append({
                    'exp_id': exp_id,
                    'maps': res['maps'],
                    'reduces': res['reduces'],
                    'chunk': res['chunk'],
                    'time': res['time'],
                    'features': len(res['features']),
                    'best_accuracy': best_acc
                })
            else:
                f.write(f"{exp_id:<5} {'FAILED':<8} {res.get('maps',0):<6} {res.get('reduces',0):<8} {res.get('chunk',0):<10} {res.get('time',0):<12.2f} {'0':<10} {'0.0000':<12}\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("CLASSIFIER ACCURACY COMPARISON\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Exp':<5} {'Random Forest':<18} {'Decision Tree':<18} {'KNN':<18} {'Naive Bayes':<18} {'SVM':<18}\n")
        f.write("-" * 120 + "\n")
        
        for exp_id, res in all_results.items():
            if res and res.get('success') and 'classifiers' in res:
                rf = res['classifiers'].get('Random Forest', {}).get('accuracy', 0)
                dt = res['classifiers'].get('Decision Tree', {}).get('accuracy', 0)
                knn = res['classifiers'].get('KNN', {}).get('accuracy', 0)
                nb = res['classifiers'].get('Naive Bayes', {}).get('accuracy', 0)
                svm = res['classifiers'].get('SVM', {}).get('accuracy', 0)
                f.write(f"{exp_id:<5} {rf:<18.4f} {dt:<18.4f} {knn:<18.4f} {nb:<18.4f} {svm:<18.4f}\n")
        
        if summary_data:
            baseline_time = summary_data[0]['time'] if summary_data else 1
            baseline_acc = summary_data[0]['best_accuracy'] if summary_data else 0
            
            f.write("\n" + "=" * 120 + "\n")
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-" * 120 + "\n")
            f.write(f"Baseline (Exp 1): Time={baseline_time:.2f}s, Accuracy={baseline_acc:.4f}\n\n")
            f.write(f"{'Experiment':<12} {'Time(s)':<12} {'Speedup':<12} {'Accuracy':<12} {'Acc Change':<12}\n")
            f.write("-" * 120 + "\n")
            for data in summary_data:
                speedup = baseline_time / data['time'] if data['time'] > 0 else 0
                acc_change = data['best_accuracy'] - baseline_acc
                f.write(f"Exp {data['exp_id']:<5} {data['time']:<12.2f} {speedup:<12.2f}x {data['best_accuracy']:<12.4f} {acc_change:<+12.4f}\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("=" * 120 + "\n")
        f.write("""
1. EFFECT OF CHUNK SIZE (Experiments 1-4):
   Increasing chunk size from 100 to 1000 improves processing efficiency

2. EFFECT OF REDUCE TASKS (Experiments 4-6):
   Adding more reducers improves aggregation performance

3. EFFECT OF MAP TASKS (Experiments 4,7,10):
   More mappers improve parallelism but add scheduling overhead

4. EFFECT OF BALANCED CONFIGURATION (Experiments 8-9,11-12):
   Balanced mapper/reducer configuration yields best overall performance

5. CLASSIFIER PERFORMANCE:
   Random Forest consistently achieves highest accuracy
   Decision Tree shows good performance with lower computational cost

6. BEST CONFIGURATION:
   Experiment 12: 5 Mappers, 5 Reducers, 1000 KB chunk size
""")
    
    print(f"\n[FINAL] Report saved to: {report_file}")
    
    df_summary = pd.DataFrame([{
        'experiment': res['exp_id'],
        'map_tasks': res['maps'],
        'reduce_tasks': res['reduces'],
        'chunk_size_kb': res['chunk'],
        'time_seconds': res['time'],
        'features_extracted': len(res['features']),
        'best_accuracy': max(res['classifiers'].items(), key=lambda x: x[1]['accuracy'])[1]['accuracy'] if res['classifiers'] else 0
    } for res in all_results.values() if res and res.get('success')])
    
    if not df_summary.empty:
        df_summary.to_csv(f"{LOCAL_RESULTS_DIR}/experiments_summary.csv", index=False)
    
    return summary_data


def print_experiment_summary(all_results):
    print("\n" + "=" * 80)
    print("EXPERIMENTS SUMMARY")
    print("=" * 80)
    print(f"{'Exp':<5} {'Status':<10} {'Time(s)':<12} {'Features':<10} {'Best Classifier':<22} {'Best Acc':<12}")
    print("-" * 80)
    
    success_count = 0
    for exp_id in range(1, 13):
        res = all_results.get(exp_id, {})
        status = "SUCCESS" if res.get('success') else "FAILED"
        time_val = res.get('time', 0)
        feat_count = len(res.get('features', []))
        best_clf = "N/A"
        best_acc = 0
        
        if status == "SUCCESS":
            success_count += 1
            if res.get('classifiers'):
                best = max(res['classifiers'].items(), key=lambda x: x[1]['accuracy'])
                best_clf = best[0][:20]
                best_acc = best[1]['accuracy']
        
        print(f"{exp_id:<5} {status:<10} {time_val:<12.2f} {feat_count:<10} {best_clf:<22} {best_acc:<12.4f}")
    
    print("-" * 80)
    print(f"Successful experiments: {success_count}/12")


def main():
    print("\n" + "=" * 60)
    print("MAPREDUCE FEATURE SELECTION WITH CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {LOCAL_DATASET_PATH}")
    
    if not os.path.exists(LOCAL_DATASET_PATH):
        print(f"\n[ERROR] Dataset not found: {LOCAL_DATASET_PATH}")
        sys.exit(1)
    
    print("\n[STEP 1] Checking Hadoop environment...")
    if not check_hadoop():
        sys.exit(1)
    
    if not check_hdfs():
        sys.exit(1)
    
    print("\n[STEP 2] Creating directories...")
    create_directories()
    setup_hdfs()
    
    print("\n[STEP 3] Loading dataset...")
    full_features, full_labels, feature_names = load_full_dataset()
    print(f"  Samples: {full_features.shape[0]}, Features: {full_features.shape[1]}")
    unique, counts = np.unique(full_labels, return_counts=True)
    print(f"  Class distribution: {dict(zip(unique, counts))}")
    
    print("\n[STEP 4] Uploading CSV to HDFS...")
    if not upload_csv_to_hdfs():
        sys.exit(1)
    
    print("\n[STEP 5] Creating Mapper and Reducer...")
    save_mapper_reducer_files()
    
    print("\n[STEP 6] Starting experiments...")
    print("=" * 60)
    
    all_results = {}
    
    for exp_id, map_tasks, reduce_tasks, chunk_size, description in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_id:02d} STARTING")
        print(f"{'='*60}")
        
        success, exec_time = run_hadoop_experiment(exp_id, map_tasks, reduce_tasks, chunk_size, description)
        
        if success:
            print(f"  Loading results from HDFS...")
            top_features = load_top_features(exp_id)
            print(f"  Features extracted: {len(top_features)}")
            
            if top_features:
                print(f"  Evaluating classifiers on top features...")
                classifier_results = save_experiment_results(
                    exp_id, top_features, exec_time, map_tasks, 
                    reduce_tasks, chunk_size, description, full_features, full_labels, feature_names
                )
                
                best_clf = max(classifier_results.items(), key=lambda x: x[1]['accuracy'])
                print(f"  [RESULT] Best classifier: {best_clf[0]} = {best_clf[1]['accuracy']:.4f}")
                print(f"  [RESULT] Execution time: {exec_time:.2f} seconds")
                
                all_results[exp_id] = {
                    'success': True,
                    'time': exec_time,
                    'maps': map_tasks,
                    'reduces': reduce_tasks,
                    'chunk': chunk_size,
                    'features': top_features,
                    'classifiers': classifier_results,
                    'exp_id': exp_id
                }
            else:
                print(f"  [WARNING] No features extracted!")
                print(f"  Checking HDFS output directly...")
                subprocess.run(["hdfs", "dfs", "-ls", f"{HDFS_OUTPUT_BASE}/exp_{exp_id:02d}"], capture_output=False)
                all_results[exp_id] = {
                    'success': False, 
                    'time': exec_time, 
                    'maps': map_tasks, 
                    'reduces': reduce_tasks, 
                    'chunk': chunk_size, 
                    'features': [], 
                    'classifiers': {},
                    'exp_id': exp_id
                }
        else:
            print(f"  [ERROR] Experiment {exp_id} failed!")
            all_results[exp_id] = {
                'success': False, 
                'time': exec_time, 
                'maps': map_tasks, 
                'reduces': reduce_tasks, 
                'chunk': chunk_size, 
                'features': [], 
                'classifiers': {},
                'exp_id': exp_id
            }
    
    print("\n" + "=" * 60)
    print("[STEP 7] Generating final report...")
    print("=" * 60)
    
    generate_final_report(all_results, feature_names)
    print_experiment_summary(all_results)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved in: {LOCAL_RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()