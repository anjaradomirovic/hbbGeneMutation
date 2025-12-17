# utils/protein_encoder.py

import pandas as pd
import numpy as np
from pathlib import Path
import pickle


class ProteinOneHotEncoder:
    """
    One-Hot Encoding za proteinske sekvence
    Priprema podatke za Machine Learning modele
    """

    def __init__(self, max_length=None):
        """
        Inicijalizacija encodera

        Args:
            max_length: Maksimalna dužina sekvence (None = automatski detektuje)
        """
        # Standardnih 20 aminokiselina + Stop kodon + Padding
        self.amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
            '*',  # Stop kodon
            'X'  # Unknown/Padding
        ]

        # Mapiranje amino kiselina na indekse
        self.aa_to_index = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.index_to_aa = {idx: aa for aa, idx in self.aa_to_index.items()}

        # Maksimalna dužina sekvence
        self.max_length = max_length

        # Broj feature-a po poziciji (22 amino kiseline)
        self.n_features = len(self.amino_acids)

        print(f"\n{'=' * 80}")
        print("PROTEIN ONE-HOT ENCODER - Inicijalizovan")
        print(f"{'=' * 80}")
        print(f"  Amino kiseline: {len(self.amino_acids)} (20 AA + Stop + Padding)")
        print(f"  Max dužina: {self.max_length if self.max_length else 'Auto-detect'}")
        print(f"  Broj feature-a po poziciji: {self.n_features}")
        print(f"{'=' * 80}\n")

    def encode_sequence(self, sequence):
        """
        One-hot enkoduje jednu proteinsku sekvencu

        Args:
            sequence (str): Proteinska sekvenca (npr. "MVHLTPEEKS...")

        Returns:
            np.array: Shape (max_length, n_features)
        """
        if not sequence:
            sequence = ""

        # Padding do max_length
        if self.max_length:
            if len(sequence) > self.max_length:
                # Skrati sekvencu
                sequence = sequence[:self.max_length]
            else:
                # Dodaj padding
                sequence = sequence + 'X' * (self.max_length - len(sequence))

        # Kreiraj one-hot matricu
        encoded = np.zeros((len(sequence), self.n_features), dtype=np.float32)

        for i, aa in enumerate(sequence):
            # Ako je nepoznata amino kiselina, označi kao 'X'
            aa_upper = aa.upper()
            if aa_upper not in self.aa_to_index:
                aa_upper = 'X'

            # Postavi odgovarajući indeks na 1
            idx = self.aa_to_index[aa_upper]
            encoded[i, idx] = 1.0

        return encoded

    def encode_batch(self, sequences):
        """
        One-hot enkoduje batch proteinskih sekvenci

        ulaz:
            sequences (list): Lista proteinskih sekvenci

        izlaz:
            np.array: Shape (n_sequences, max_length, n_features)
        """
        # Detektuj max_length ako nije setovan
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences if seq)
            print(f"  Auto-detected max_length: {self.max_length}")

        # Enkoduj sve sekvence
        encoded_sequences = []
        for seq in sequences:
            encoded = self.encode_sequence(seq)
            encoded_sequences.append(encoded)

        return np.array(encoded_sequences)

    def flatten_encoding(self, encoded_sequence):
        """
        Spljošti one-hot encoding u 1D vektor

        Args:
            encoded_sequence: Shape (max_length, n_features)

        Returns:
            np.array: Shape (max_length * n_features,)
        """
        return encoded_sequence.flatten()

    def process_protein_file(self,
                             input_file='data/processed/protein_translations.csv',
                             output_file='data/ml_ready/encoded_proteins.npz',
                             include_reference=True):
        """
        Procesira CSV sa proteinima i generiše enkodovane podatke za ML

        Args:
            input_file: CSV sa protein translacijama
            output_file: NPZ fajl sa enkodovanim podacima
            include_reference: Da li enkodovati i referentni protein

        Returns:
            dict sa svim podacima spremnim za ML
        """
        print(f"\n{'=' * 80}")
        print("PROTEIN ENCODING PIPELINE")
        print(f"{'=' * 80}\n")

        # 1. Učitaj podatke
        print(f"[1] Učitavanje podataka iz {input_file}...")
        df = pd.read_csv(input_file)
        print(f"    ✓ Učitano {len(df)} mutacija")

        # 2. Filtriraj validne proteine
        valid = df[
            (df['mutated_protein'].notna()) &
            (df['mutated_protein'] != 'None') &
            (df['translation_status'] == 'SUCCESS')
            ].copy()

        print(f"    ✓ Validnih proteina: {len(valid)}")

        if len(valid) == 0:
            print("\n    ⚠️  Nema validnih proteina za enkodovanje!")
            return None

        # 3. Ekstraktuj sekvence
        print(f"\n[2] Ekstrakcija proteinskih sekvenci...")

        reference_proteins = valid['reference_protein'].tolist()
        mutated_proteins = valid['mutated_protein'].tolist()

        # Detektuj max dužinu
        all_sequences = mutated_proteins + reference_proteins
        self.max_length = max(len(str(seq)) for seq in all_sequences if pd.notna(seq))
        print(f"    ✓ Max dužina proteina: {self.max_length} AA")

        # 4. One-hot enkodovanje
        print(f"\n[3] One-hot enkodovanje...")
        print(f"    Enkodujem {len(mutated_proteins)} mutiranih proteina...")

        # Enkoduj mutirane proteine
        mutated_encoded = self.encode_batch([str(p) if pd.notna(p) else "" for p in mutated_proteins])
        print(f"    ✓ Mutated shape: {mutated_encoded.shape}")

        # Enkoduj referentne proteine (ako je potrebno)
        reference_encoded = None
        if include_reference:
            print(f"    Enkodujem {len(reference_proteins)} referentnih proteina...")
            reference_encoded = self.encode_batch([str(p) if pd.notna(p) else "" for p in reference_proteins])
            print(f"    ✓ Reference shape: {reference_encoded.shape}")

        # 5. Pripremi labels (Clinical Significance)
        print(f"\n[4] Priprema label-a...")

        # Mapiranje clinical significance u numeričke labele
        label_mapping = {
            'Pathogenic': 1,
            'Likely pathogenic': 1,
            'Pathogenic/Likely pathogenic': 1,
            'Benign': 0,
            'Likely benign': 0,
            'Benign/Likely benign': 0,
            'Uncertain significance': 2,
            'Conflicting interpretations of pathogenicity': 2,
            'not provided': 2,
            'drug response': 3,
            'association': 3,
            'risk factor': 3,
            'Affects': 3,
            'protective': 3
        }

        # Mapiraj labels
        labels = valid['clinical_significance'].map(label_mapping).fillna(2).astype(int)
        labels_array = labels.values

        print(f"    ✓ Labels: {len(labels_array)}")
        print(f"    Distribucija:")
        label_counts = pd.Series(labels_array).value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = {0: 'Benign', 1: 'Pathogenic', 2: 'Uncertain', 3: 'Other'}[label]
            print(f"      {label_name} ({label}): {count}")

        # 6. Dodatni feature-i
        print(f"\n[5] Ekstrakcija dodatnih feature-a...")

        additional_features = valid[[
            'protein_length_change',
            'amino_acid_changes',
            'has_premature_stop',
            'is_truncated',
            'is_extended',
            'splice_affected',
            'splice_score'
        ]].fillna(0)

        # Konvertuj boolean u int
        additional_features['has_premature_stop'] = additional_features['has_premature_stop'].astype(int)
        additional_features['is_truncated'] = additional_features['is_truncated'].astype(int)
        additional_features['is_extended'] = additional_features['is_extended'].astype(int)
        additional_features['splice_affected'] = additional_features['splice_affected'].astype(int)

        additional_features_array = additional_features.values.astype(np.float32)
        print(f"    ✓ Dodatnih feature-a: {additional_features_array.shape[1]}")

        # 7. Metadata
        metadata = valid[[
            'variant_id',
            'chromosome',
            'position',
            'ref_allele',
            'alt_allele',
            'hgvs',
            'clinical_significance',
            'location_type',
            'location_detail'
        ]].copy()

        # 8. Sačuvaj sve podatke
        print(f"\n[6] Čuvanje enkodovanih podataka...")

        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sačuvaj u NPZ format (kompresovano)
        save_dict = {
            'mutated_proteins_encoded': mutated_encoded,
            'labels': labels_array,
            'additional_features': additional_features_array,
            'feature_names': additional_features.columns.tolist(),
            'label_mapping': label_mapping,
            'max_length': self.max_length,
            'n_features_per_position': self.n_features,
            'n_samples': len(valid)
        }

        if include_reference:
            save_dict['reference_proteins_encoded'] = reference_encoded

        np.savez_compressed(output_file, **save_dict)
        print(f"    ✓ Sačuvano u: {output_file}")

        # Sačuvaj metadata kao CSV
        metadata_file = output_file.replace('.npz', '_metadata.csv')
        metadata.to_csv(metadata_file, index=False)
        print(f"    ✓ Metadata sačuvana u: {metadata_file}")

        # Sačuvaj encoder
        encoder_file = output_file.replace('.npz', '_encoder.pkl')
        with open(encoder_file, 'wb') as f:
            pickle.dump(self, f)
        print(f"    ✓ Encoder sačuvan u: {encoder_file}")

        # 9. Statistika
        print(f"\n{'=' * 80}")
        print("STATISTIKA ENKODOVANJA")
        print(f"{'=' * 80}")
        print(f"Dataset:")
        print(f"  Broj uzoraka:              {len(valid)}")
        print(f"  Max dužina proteina:       {self.max_length} AA")
        print(f"\nShape-ovi:")
        print(f"  Mutated proteins:          {mutated_encoded.shape}")
        if include_reference:
            print(f"  Reference proteins:        {reference_encoded.shape}")
        print(f"  Labels:                    {labels_array.shape}")
        print(f"  Additional features:       {additional_features_array.shape}")
        print(f"\nFeature-i:")
        print(f"  One-hot per position:      {self.n_features}")
        print(f"  Total one-hot features:    {self.max_length * self.n_features}")
        print(f"  Additional features:       {additional_features_array.shape[1]}")
        print(f"  TOTAL features:            {self.max_length * self.n_features + additional_features_array.shape[1]}")
        print(f"\n{'=' * 80}")
        print(f"✓ Podaci spremni za Machine Learning!")
        print(f"{'=' * 80}\n")

        return {
            'mutated_encoded': mutated_encoded,
            'reference_encoded': reference_encoded,
            'labels': labels_array,
            'additional_features': additional_features_array,
            'feature_names': additional_features.columns.tolist(),
            'metadata': metadata,
            'encoder': self
        }

    def load_encoded_data(self, npz_file='data/ml_ready/encoded_proteins.npz'):
        """
        Učitava enkodovane podatke iz NPZ fajla

        Returns:
            dict sa svim podacima
        """
        print(f"Učitavanje enkodovanih podataka iz {npz_file}...")

        data = np.load(npz_file, allow_pickle=True)

        result = {
            'mutated_encoded': data['mutated_proteins_encoded'],
            'labels': data['labels'],
            'additional_features': data['additional_features'],
            'feature_names': data['feature_names'].tolist(),
            'label_mapping': data['label_mapping'].item(),
            'max_length': data['max_length'].item(),
            'n_features_per_position': data['n_features_per_position'].item()
        }

        if 'reference_proteins_encoded' in data:
            result['reference_encoded'] = data['reference_proteins_encoded']

        print(f"✓ Učitano {result['mutated_encoded'].shape[0]} uzoraka")

        return result


# ========== HELPER FUNKCIJE ZA ML ==========

def prepare_for_ml(encoded_data, flatten=True, include_additional=True):
    """
    Priprema enkodovane podatke u format za ML algoritme

    Args:
        encoded_data: Dict iz ProteinOneHotEncoder.process_protein_file()
        flatten: Da li spljoštiti one-hot encoding u 1D
        include_additional: Da li uključiti dodatne feature-e

    Returns:
        X, y: Feature matrica i labele
    """
    mutated_encoded = encoded_data['mutated_encoded']
    labels = encoded_data['labels']
    additional_features = encoded_data['additional_features']

    # Flatten one-hot encoding
    if flatten:
        n_samples = mutated_encoded.shape[0]
        X_onehot = mutated_encoded.reshape(n_samples, -1)
        print(f"One-hot flattened shape: {X_onehot.shape}")
    else:
        X_onehot = mutated_encoded

    # Dodaj dodatne feature-e
    if include_additional:
        X = np.concatenate([X_onehot, additional_features], axis=1)
        print(f"Final X shape (with additional features): {X.shape}")
    else:
        X = X_onehot

    y = labels

    return X, y


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Kreira train/test split
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Održava proporciju klasa
    )

    print(f"\nTrain/Test Split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test