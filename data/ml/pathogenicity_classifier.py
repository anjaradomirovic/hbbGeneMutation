# ml/pathogenicity_classifier.py - POPRAVLJENA VERZIJA

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)

# ML modeli
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# SMOTE za balansiranje
from imblearn.over_sampling import SMOTE
from collections import Counter

import warnings

warnings.filterwarnings('ignore')


class PathogenicityClassifier:
    """
    ML sistem za predikciju patogenosti mutacija - POPRAVLJENA VERZIJA
    """

    def __init__(self, random_state=42, use_smote=True):
        """
        Inicijalizacija klasifikatora

        Args:
            random_state: Seed za reproducibilnost
            use_smote: Da li koristiti SMOTE za balansiranje klasa
        """
        self.random_state = random_state
        self.use_smote = use_smote
        self.scaler = StandardScaler()
        self.smote = None

        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0

        self.models = {}
        self.scores = {}

        # Label encoding
        self.label_map = {
            0: 'Benign',
            1: 'Pathogenic',
            2: 'Uncertain',
            3: 'Other'
        }

        print(f"\n{'=' * 80}")
        print("PATHOGENICITY CLASSIFIER - POPRAVLJENA VERZIJA")
        print(f"{'=' * 80}")
        print(f"  Random state: {random_state}")
        print(f"  SMOTE balansiranje: {'✓ DA' if use_smote else '✗ NE'}")
        print(f"  Label mapping: {self.label_map}")
        print(f"{'=' * 80}\n")

    def filter_clean_labels(self, X, y, verbose=True):
        """
        🔧 NOVA FUNKCIJA: Filtrira samo čiste Pathogenic/Benign labele

        Args:
            X: Feature matrica
            y: Labels (0=Benign, 1=Pathogenic, 2=Uncertain, 3=Other)
            verbose: Print statistiku

        Returns:
            X_clean, y_clean: Filtrirani podaci bez Uncertain/Other
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print("🧹 ČIŠĆENJE LABELA")
            print(f"{'=' * 80}")
            print(f"\n📊 Pre čišćenja:")
            print(f"  Ukupno uzoraka: {len(y)}")
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                label = self.label_map.get(u, f'Unknown({u})')
                print(f"    {label}: {c} ({c / len(y) * 100:.1f}%)")

        # Filtriraj samo Benign (0) i Pathogenic (1)
        clean_mask = (y == 0) | (y == 1)
        X_clean = X[clean_mask]
        y_clean = y[clean_mask]

        if verbose:
            print(f"\n✅ Posle čišćenja:")
            print(f"  Ukupno uzoraka: {len(y_clean)}")
            unique, counts = np.unique(y_clean, return_counts=True)
            for u, c in zip(unique, counts):
                label = self.label_map.get(u, f'Unknown({u})')
                print(f"    {label}: {c} ({c / len(y_clean) * 100:.1f}%)")

            removed = len(y) - len(y_clean)
            print(f"\n  ❌ Uklonjeno: {removed} uzoraka ({removed / len(y) * 100:.1f}%)")
            print(f"  ✓ Zadržano: {len(y_clean)} uzoraka ({len(y_clean) / len(y) * 100:.1f}%)")
            print(f"{'=' * 80}\n")

        return X_clean, y_clean

    def get_models(self, use_class_weight=True):
        """
        Definiše sve ML modele sa class_weight='balanced'
        """
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced' if use_class_weight else None,
                random_state=self.random_state,
                n_jobs=-1
            ),

            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3 if use_class_weight else 1,  # Za imbalance
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
            ),

            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                random_state=self.random_state
            ),

            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                random_state=self.random_state
            ),

            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced' if use_class_weight else None,
                probability=True,
                random_state=self.random_state
            ),

            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight='balanced' if use_class_weight else None,
                random_state=self.random_state,
                n_jobs=-1
            ),

            'Naive Bayes': GaussianNB()
        }

        return models

    def apply_smote(self, X_train, y_train):
        """
        Primenjuje SMOTE za balansiranje klasa
        """
        print(f"\n{'=' * 80}")
        print("⚖️  SMOTE BALANSIRANJE")
        print(f"{'=' * 80}")

        print(f"\n📊 Pre SMOTE:")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            label = 'Pathogenic' if u == 1 else 'Benign'
            print(f"  {label}: {c} ({c / len(y_train) * 100:.1f}%)")

        # SMOTE
        self.smote = SMOTE(random_state=self.random_state, k_neighbors=3)
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)

        print(f"\n✅ Posle SMOTE:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for u, c in zip(unique, counts):
            label = 'Pathogenic' if u == 1 else 'Benign'
            print(f"  {label}: {c} ({c / len(y_train_balanced) * 100:.1f}%)")

        print(f"\n  📈 Povećanje: {len(y_train)} → {len(y_train_balanced)} uzoraka")
        print(f"{'=' * 80}\n")

        return X_train_balanced, y_train_balanced

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Trenira i evaluira sve modele - POPRAVLJENA VERZIJA
        """
        print(f"\n{'=' * 80}")
        print("TRENIRANJE I EVALUACIJA MODELA")
        print(f"{'=' * 80}\n")

        # Skaliraj feature-e PRVO (pre SMOTE!)
        print(f"[1] Skaliranje feature-a...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"    ✓ Feature-i skalirani")

        # Primeni SMOTE AKO JE OMOGUĆEN
        if self.use_smote:
            X_train_scaled, y_train = self.apply_smote(X_train_scaled, y_train)

        # Treniranje modela
        print(f"\n[2] Treniranje modela...\n")

        models = self.get_models(use_class_weight=True)
        results = []

        for name, model in models.items():
            print(f"  Treniram {name}...", end=" ")

            try:
                # Treniranje
                model.fit(X_train_scaled, y_train)

                # Predikcije
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None

                # Metrike
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

                # ROC-AUC
                roc_auc = None
                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                # Sačuvaj model i rezultate
                self.models[name] = model
                self.scores[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }

                results.append({
                    'Model': name,
                    'Accuracy': f"{accuracy:.4f}",
                    'Precision': f"{precision:.4f}",
                    'Recall': f"{recall:.4f}",
                    'F1-Score': f"{f1:.4f}",
                    'ROC-AUC': f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
                    'CV F1 (mean±std)': f"{cv_mean:.4f}±{cv_std:.4f}"
                })

                # Ažuriraj najbolji model
                if f1 > self.best_score:
                    self.best_score = f1
                    self.best_model = model
                    self.best_model_name = name

                # Popravljena linija printa
                roc_display = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
                print(f"✓ F1: {f1:.4f}, Recall: {recall:.4f}, ROC: {roc_display}")

            except Exception as e:
                print(f"✗ Greška: {e}")
                continue

        # Prikaz rezultata
        print(f"\n{'=' * 80}")
        print("REZULTATI SVIH MODELA")
        print(f"{'=' * 80}\n")

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        print(f"\n{'=' * 80}")
        print(f"🏆 NAJBOLJI MODEL: {self.best_model_name}")
        print(f"{'=' * 80}")
        print(f"  F1-Score: {self.best_score:.4f}")
        print(f"  Recall:   {self.scores[self.best_model_name]['recall']:.4f}")
        print(f"  ROC-AUC:  {self.scores[self.best_model_name]['roc_auc']:.4f}")
        print(
            f"  CV Score: {self.scores[self.best_model_name]['cv_mean']:.4f} ± {self.scores[self.best_model_name]['cv_std']:.4f}")
        print(f"{'=' * 80}\n")

        return {
            'results_df': results_df,
            'best_model_name': self.best_model_name,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'y_test': y_test
        }

    def plot_results(self, evaluation_results, output_dir='data/ml_results'):
        """Plotuje rezultate - ISTO kao pre"""
        print(f"\n[3] Generisanje vizualizacija...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        best_name = evaluation_results['best_model_name']
        y_test = evaluation_results['y_test']
        y_pred = self.scores[best_name]['y_pred']
        y_pred_proba = self.scores[best_name]['y_pred_proba']

        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Pathogenic'],
                    yticklabels=['Benign', 'Pathogenic'])
        plt.title(f'Confusion Matrix - {best_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
        plt.close()
        print(f"    ✓ Confusion matrix: {output_dir}/confusion_matrix.png")

        # 2. ROC Curve
        if y_pred_proba is not None:
            plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'{best_name} (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Pathogenicity Prediction')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/roc_curve.png', dpi=300)
            plt.close()
            print(f"    ✓ ROC curve: {output_dir}/roc_curve.png")

        # 3. Model Comparison
        results_df = evaluation_results['results_df']
        plt.figure(figsize=(14, 8))

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(results_df))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = results_df[metric].astype(float)
            plt.bar(x + i * width, values, width, label=metric)

        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 1.5, results_df['Model'], rotation=45, ha='right')
        plt.ylim([0, 1.05])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300)
        plt.close()
        print(f"    ✓ Model comparison: {output_dir}/model_comparison.png")

    def generate_classification_report(self, evaluation_results,
                                       output_file='data/ml_results/classification_report.txt'):
        """Generiše report"""
        best_name = evaluation_results['best_model_name']
        y_test = evaluation_results['y_test']
        y_pred = self.scores[best_name]['y_pred']

        report = classification_report(y_test, y_pred,
                                       target_names=['Benign', 'Pathogenic'])

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(f"CLASSIFICATION REPORT - {best_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            f.write("\n\nCONFUSION MATRIX:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))

        print(f"\n{'=' * 80}")
        print(f"DETAILED CLASSIFICATION REPORT - {best_name}")
        print(f"{'=' * 80}\n")
        print(report)
        print(f"\n✓ Detaljan report sačuvan u: {output_file}")

    def save_best_model(self, output_file='data/ml_results/best_model.pkl'):
        """Čuva model"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'smote': self.smote,
            'score': self.best_score,
            'label_map': self.label_map,
            'use_smote': self.use_smote
        }

        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Najbolji model sačuvan u: {output_file}")


# ========== GLAVNI PIPELINE - POPRAVLJEN ==========

def train_pathogenicity_model(encoded_data_file='data/ml_ready/encoded_proteins.npz',
                              output_dir='data/ml_results',
                              use_smote=True,
                              filter_uncertain=True):
    """
    Kompletan pipeline - POPRAVLJENA VERZIJA

    Args:
        encoded_data_file: NPZ fajl sa enkodovanim podacima
        output_dir: Folder za rezultate
        use_smote: Da li koristiti SMOTE balansiranje
        filter_uncertain: Da li ukloniti Uncertain labele (PREPORUKA: True!)
    """
    print(f"\n{'=' * 80}")
    print("PATHOGENICITY MODEL TRAINING PIPELINE - POPRAVLJENA VERZIJA")
    print(f"{'=' * 80}\n")

    # 1. Učitaj podatke
    print(f"[1] Učitavanje enkodovanih podataka...")
    data = np.load(encoded_data_file, allow_pickle=True)

    mutated_encoded = data['mutated_proteins_encoded']
    labels = data['labels']
    additional_features = data['additional_features']

    print(f"    ✓ Učitano {mutated_encoded.shape[0]} uzoraka")

    # 2. Pripremi feature-e
    print(f"\n[2] Priprema feature-a...")
    n_samples = mutated_encoded.shape[0]
    X_onehot = mutated_encoded.reshape(n_samples, -1)
    X = np.concatenate([X_onehot, additional_features], axis=1)
    y = labels

    print(f"    ✓ Feature shape: {X.shape}")
    print(f"    ✓ Label shape: {y.shape}")

    # 3. Inicijalizuj classifier
    classifier = PathogenicityClassifier(random_state=42, use_smote=use_smote)

    # 4. 🔧 ČISTI LABELE (ukloni Uncertain/Other)
    if filter_uncertain:
        X, y = classifier.filter_clean_labels(X, y, verbose=True)

    # 5. Train/Test split
    print(f"\n[3] Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"    ✓ Train: {X_train.shape[0]} samples")
    print(f"    ✓ Test:  {X_test.shape[0]} samples")

    # 6. Treniraj i evaluiraj modele
    results = classifier.train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # 7. Vizualizacije
    classifier.plot_results(results, output_dir=output_dir)

    # 8. Detaljan report
    classifier.generate_classification_report(results,
                                              output_file=f'{output_dir}/classification_report.txt')

    # 9. Sačuvaj najbolji model
    classifier.save_best_model(output_file=f'{output_dir}/best_model.pkl')

    print(f"\n{'=' * 80}")
    print("✓ TRENIRANJE ZAVRŠENO")
    print(f"{'=' * 80}")
    print(f"\n📁 Rezultati:")
    print(f"   - Model: {output_dir}/best_model.pkl")
    print(f"   - Report: {output_dir}/classification_report.txt")
    print(f"   - Plots: {output_dir}/*.png")
    print(f"{'=' * 80}\n")

    return classifier, results