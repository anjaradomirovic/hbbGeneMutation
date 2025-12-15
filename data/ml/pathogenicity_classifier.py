# ml/pathogenicity_classifier.py

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)

# ML modeli
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')


class PathogenicityClassifier:
    """
    ML sistem za predikciju patogenosti mutacija
    """
    
    def __init__(self, random_state=42):
        """
        Inicijalizacija klasifikatora
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Najbolji model (nakon treniranja)
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        
        # Čuvanje svih modela
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
        print("PATHOGENICITY CLASSIFIER - Inicijalizovan")
        print(f"{'=' * 80}")
        print(f"  Random state: {random_state}")
        print(f"  Label mapping: {self.label_map}")
        print(f"{'=' * 80}\n")
    
    def get_models(self):
        """
        Definiše sve ML modele za testiranje
        """
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='mlogloss',
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
                probability=True,
                random_state=self.random_state
            ),
            
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Naive Bayes': GaussianNB()
        }
        
        return models
    
    def prepare_binary_classification(self, y):
        """
        Konvertuje multi-class u binary classification (Pathogenic vs Not Pathogenic)
        
        Args:
            y: Labels (0=Benign, 1=Pathogenic, 2=Uncertain, 3=Other)
        
        Returns:
            y_binary: (0=Not Pathogenic, 1=Pathogenic)
        """
        # 1 = Pathogenic, ostalo = Not Pathogenic (0)
        y_binary = (y == 1).astype(int)
        return y_binary
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, 
                                  binary_classification=True):
        """
        Trenira i evaluira sve modele
        
        Args:
            X_train, X_test: Feature matrice
            y_train, y_test: Labels
            binary_classification: Da li koristiti binarnu klasifikaciju
        
        Returns:
            dict sa rezultatima
        """
        print(f"\n{'=' * 80}")
        print("TRENIRANJE I EVALUACIJA MODELA")
        print(f"{'=' * 80}\n")
        
        # Binary classification (Pathogenic vs Not Pathogenic)
        if binary_classification:
            print("📊 Mod: BINARY CLASSIFICATION (Pathogenic vs Not Pathogenic)")
            y_train_proc = self.prepare_binary_classification(y_train)
            y_test_proc = self.prepare_binary_classification(y_test)
            
            print(f"\nDistribucija Train seta:")
            unique, counts = np.unique(y_train_proc, return_counts=True)
            for u, c in zip(unique, counts):
                label = 'Pathogenic' if u == 1 else 'Not Pathogenic'
                print(f"  {label}: {c} ({c/len(y_train_proc)*100:.1f}%)")
        else:
            print("📊 Mod: MULTI-CLASS CLASSIFICATION")
            y_train_proc = y_train
            y_test_proc = y_test
        
        # Skaliraj feature-e
        print(f"\n[1] Skaliranje feature-a...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"    ✓ Feature-i skalirani")
        
        # Treniranje modela
        print(f"\n[2] Treniranje modela...\n")
        
        models = self.get_models()
        results = []
        
        for name, model in models.items():
            print(f"  Treniram {name}...", end=" ")
            
            try:
                # Treniranje
                model.fit(X_train_scaled, y_train_proc)
                
                # Predikcije
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Metrike
                accuracy = accuracy_score(y_test_proc, y_pred)
                precision = precision_score(y_test_proc, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test_proc, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test_proc, y_pred, average='weighted', zero_division=0)
                
                # ROC-AUC (samo za binary)
                roc_auc = None
                if binary_classification and y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test_proc, y_pred_proba[:, 1])
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train_proc, 
                    cv=5, 
                    scoring='f1_weighted',
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
                    'ROC-AUC': f"{roc_auc:.4f}" if roc_auc else "N/A",
                    'CV F1 (mean±std)': f"{cv_mean:.4f}±{cv_std:.4f}"
                })
                
                # Ažuriraj najbolji model
                if f1 > self.best_score:
                    self.best_score = f1
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"✓ F1: {f1:.4f}, CV: {cv_mean:.4f}±{cv_std:.4f}")
                
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
        print(f"  CV Score: {self.scores[self.best_model_name]['cv_mean']:.4f} ± {self.scores[self.best_model_name]['cv_std']:.4f}")
        if binary_classification:
            print(f"  ROC-AUC:  {self.scores[self.best_model_name]['roc_auc']:.4f}")
        print(f"{'=' * 80}\n")
        
        return {
            'results_df': results_df,
            'best_model_name': self.best_model_name,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'y_test': y_test_proc,
            'binary': binary_classification
        }
    
    def plot_results(self, evaluation_results, output_dir='data/ml_results'):
        """
        Plotuje rezultate modela
        """
        print(f"\n[3] Generisanje vizualizacija...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        best_name = evaluation_results['best_model_name']
        y_test = evaluation_results['y_test']
        y_pred = self.scores[best_name]['y_pred']
        y_pred_proba = self.scores[best_name]['y_pred_proba']
        binary = evaluation_results['binary']
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        
        if binary:
            labels = ['Not Pathogenic', 'Pathogenic']
        else:
            labels = [self.label_map[i] for i in sorted(np.unique(y_test))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {best_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
        plt.close()
        print(f"    ✓ Confusion matrix: {output_dir}/confusion_matrix.png")
        
        # 2. ROC Curve (samo za binary)
        if binary and y_pred_proba is not None:
            plt.figure(figsize=(10, 8))
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
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
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*1.5, results_df['Model'], rotation=45, ha='right')
        plt.ylim([0, 1.05])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300)
        plt.close()
        print(f"    ✓ Model comparison: {output_dir}/model_comparison.png")
        
        print(f"    ✓ Sve vizualizacije sačuvane u: {output_dir}/")
    
    def generate_classification_report(self, evaluation_results, output_file='data/ml_results/classification_report.txt'):
        """
        Generiše detaljan classification report
        """
        best_name = evaluation_results['best_model_name']
        y_test = evaluation_results['y_test']
        y_pred = self.scores[best_name]['y_pred']
        binary = evaluation_results['binary']
        
        if binary:
            target_names = ['Not Pathogenic', 'Pathogenic']
        else:
            target_names = [self.label_map[i] for i in sorted(np.unique(y_test))]
        
        report = classification_report(y_test, y_pred, target_names=target_names)
        
        # Sačuvaj u fajl
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
        """
        Čuva najbolji model
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'score': self.best_score,
            'label_map': self.label_map
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Najbolji model sačuvan u: {output_file}")
    
    def predict_pathogenicity(self, X, return_proba=True):
        """
        Predikcija patogenosti za nove podatke
        
        Args:
            X: Feature matrica
            return_proba: Da li vratiti verovatnoće
        
        Returns:
            predictions, probabilities (opciono)
        """
        if self.best_model is None:
            raise ValueError("Model nije treniran! Prvo pokreni train_and_evaluate_models()")
        
        # Skaliraj
        X_scaled = self.scaler.transform(X)
        
        # Predikcija
        predictions = self.best_model.predict(X_scaled)
        
        if return_proba and hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_scaled)
            return predictions, probabilities
        
        return predictions


# ========== GLAVNI TRENING PIPELINE ==========

def train_pathogenicity_model(encoded_data_file='data/ml_ready/encoded_proteins.npz',
                              output_dir='data/ml_results',
                              binary_classification=True):
    """
    Kompletan pipeline za treniranje modela patogenosti
    """
    print(f"\n{'=' * 80}")
    print("PATHOGENICITY MODEL TRAINING PIPELINE")
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
    
    # Flatten one-hot encoding
    n_samples = mutated_encoded.shape[0]
    X_onehot = mutated_encoded.reshape(n_samples, -1)
    
    # Kombinuj sa dodatnim feature-ima
    X = np.concatenate([X_onehot, additional_features], axis=1)
    y = labels
    
    print(f"    ✓ Feature shape: {X.shape}")
    print(f"    ✓ Label shape: {y.shape}")
    
    # 3. Train/Test split
    print(f"\n[3] Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"    ✓ Train: {X_train.shape[0]} samples")
    print(f"    ✓ Test:  {X_test.shape[0]} samples")
    
    # 4. Inicijalizuj classifier
    classifier = PathogenicityClassifier(random_state=42)
    
    # 5. Treniraj i evaluiraj modele
    results = classifier.train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        binary_classification=binary_classification
    )
    
    # 6. Vizualizacije
    classifier.plot_results(results, output_dir=output_dir)
    
    # 7. Detaljan report
    classifier.generate_classification_report(
        results, 
        output_file=f'{output_dir}/classification_report.txt'
    )
    
    # 8. Sačuvaj najbolji model
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