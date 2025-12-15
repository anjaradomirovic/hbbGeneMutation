import json
from pathlib import Path

import Bio
import pandas
import pandas as pd
import sklearn
import tensorflow
import matplotlib.pyplot as plt

from data.ml.pathogenicity_classifier import train_pathogenicity_model
from utils.hbb_mutation_processor import HBBMutationProcessor
from utils.intron_obrada import HBBStructure
from utils.proteinTranslator import ProteinTranslator
from utils.protein_encoder import prepare_for_ml, ProteinOneHotEncoder, create_train_test_split
from utils.sequence_tools import SequenceTools
import os


def main():

    # Inicijalizacija alata
    seq_tools = SequenceTools("aradomirovic1824rn@raf.rs")

    print("\n[KORAK 1] Preuzimanje HBB strukture (eksoni/introni)...")

    hbb_fetcher = HBBStructure(email = "aradomirovic1824rn@raf.rs")
    structure = hbb_fetcher.preuzmi_iz_ensembl()

    if structure is None:
        print("❌ Greška pri preuzimanju strukture!")
        return

    print("\n Struktura uspešno preuzeta!")

    # Prikaži strukturu
    print("HBB STRUKTURA - PREGLED")

    print(f"\n Gen:")
    print(f"   Simbol: {structure['gene']['symbol']}")
    print(f"   Hromozom: {structure['gene']['chromosome']}")
    print(f"   Pozicija: {structure['gene']['start']:,} - {structure['gene']['end']:,}")
    print(f"   Strand: {structure['gene']['strand']}")

    print(f"\n Eksoni ({len(structure['exons'])} ukupno):")
    for exon in structure['exons']:
        print(f"   Exon {exon['number']}: {exon['start']:,} - {exon['end']:,} ({exon['length']:,} bp)")

    print(f"\n Introni ({len(structure['introns'])} ukupno):")
    for intron in structure['introns']:
        print(f"   Intron {intron['number']}: {intron['start']:,} - {intron['end']:,} ({intron['length']:,} bp)")

    # Sačuvaj strukturu u JSON
    structure_file = "data/raw/hbb_structure.json"
    with open(structure_file, 'w') as f:
        json.dump(structure, f, indent=2)
    print(f"\n✓ Struktura sačuvana u: {structure_file}")


    # Preuzmi referentnu HBB sekvencu
    reference = seq_tools.preuzmi_hbb_referentni("data/raw/hbb_reference.fasta")

    # 2. SAMO HBB mutacije (ne skida ceo VCF!)
    csv_path = "data/raw/clinvar_hbb.csv"

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        print(f"CSV fajl već postoji i nije prazan: {csv_path}")
        mutations_df = pd.read_csv(csv_path)
    else:
        # CSV ne postoji ili je prazan, pokrećemo preuzimanje
        mutations_df = seq_tools.preuzmi_clinvar_hbb_vcf(
            output_file=csv_path  
        )

    print(f"Preuzeto {len(mutations_df)} HBB mutacija")

    # Test: Proveri šta ima u CSV-u
    if Path("data/processed/mutation_with_sequences.csv").exists():
        df_test = pd.read_csv("data/processed/mutation_with_sequences.csv")

        print("\n🔍 CSV DIJAGNOSTIKA:")
        print(f"  Redova: {len(df_test)}")

        if 'processing_status' in df_test.columns:
            processed_count = (df_test['processing_status'] == 'PROCESSED').sum()
            print(f"  PROCESSED redova: {processed_count}")

            if 'mutated_mrna' in df_test.columns:
                valid_mrna = (
                        df_test['mutated_mrna'].notna() &
                        (df_test['mutated_mrna'].astype(str).str.len() > 10)
                ).sum()
                print(f"  Validnih mRNA: {valid_mrna}")

                # Ako ima PROCESSED ali nema mRNA, to je problem!
                if processed_count > 0 and valid_mrna == 0:
                    print("  ⚠️ PROBLEM: Status je PROCESSED ali nema mRNA podataka!")
            else:
                print("  ❌ Kolona 'mutated_mrna' ne postoji")
        else:
            print("  ❌ Kolona 'processing_status' ne postoji")

        print()

    # Generisanje mutiranih nukleotidnih lanaca
    print("\n HBBMutationProcessor")

    try:
        # Inicijalizuj procesor
        processor = HBBMutationProcessor(
            structure_file="data/raw/hbb_structure.json",
            fasta_file="data/raw/hbb_reference.fasta"
        )

        # Procesiraj mutacije - SA PRESKAKANJEM VEĆ OBRAĐENIH
        mutations_with_mrna, stats = processor.process_clinvar_file(
            input_file="data/raw/clinvar_hbb.csv",
            output_file="data/processed/mutation_with_sequences.csv",
            skip_processed=True  # ✅ DODATO - preskače već obrađene
        )

        # Ispravljena statistika - koristi 'newly_processed' umesto 'processed'
        total_processed = stats['already_processed'] + stats['newly_processed']
        print(f"\n✓ Ukupno procesuirano: {total_processed}/{stats['total']}")
        print(f"  → Već postojalo: {stats['already_processed']}")
        print(f"  → Novo obrađeno: {stats['newly_processed']}")
        print(f"✓ Rezultati sačuvani u: data/processed/mutation_with_sequences.csv")

        # Prikaži primer procesuiranih mutacija (samo NOVO procesuiranih)
        if stats['newly_processed'] > 0:
            print("\n" + "-" * 80)
            print("PRIMER NOVO PROCESUIRANIH MUTACIJA:")
            print("-" * 80)

            # Filtriraj samo PROCESSED status
            processed = mutations_with_mrna[
                mutations_with_mrna['processing_status'] == 'PROCESSED'
                ].tail(3)  # Uzmi poslednje 3 (najskorije obrađene)

            for idx, row in processed.iterrows():
                print(f"\n  Mutacija #{int(row.get('original_index', idx)) + 1}:")
                print(f"    Pozicija: {row['position']}")
                print(f"    Ref→Alt: {row['ref_allele']}→{row['alt_allele']}")
                print(f"    Lokacija: {row['location_detail']}")

                # Prikaz splice info ako postoji
                if row.get('splice_affected'):
                    print(f"    Splice: ✓ Utiče (score: {row['splice_score']:.3f}, type: {row['splice_type']})")

                mrna = row['mutated_mrna']
                if mrna and len(str(mrna)) > 60:
                    print(f"    mRNA: {str(mrna)[:30]}...{str(mrna)[-30:]}")
                else:
                    print(f"    mRNA: {mrna}")

        elif stats['already_processed'] > 0:
            print(f"\n✓ Sve mutacije su već bile obrađene - nema novih.")

        # Statistika preskočenih
        if stats['skipped'] > 0:
            print(f"\n⚠ {stats['skipped']} mutacija preskočeno:")
            skipped = mutations_with_mrna[
                mutations_with_mrna['processing_status'] == 'SKIPPED'
                ]
            skip_reasons = skipped['skip_reason'].value_counts()
            for reason, count in skip_reasons.items():
                print(f"    • {reason}: {count}")

        # Statistika grešaka
        if stats['errors'] > 0:
            print(f"\n❌ {stats['errors']} mutacija sa greškama:")
            errors = mutations_with_mrna[
                mutations_with_mrna['processing_status'] == 'ERROR'
                ]
            error_reasons = errors['skip_reason'].value_counts()
            for reason, count in error_reasons.items():
                print(f"    • {reason}: {count}")

    except Exception as e:
        print(f"\n❌ GREŠKA pri procesiranju mutacija: {e}")
        import traceback
        traceback.print_exc()
        return

    # Ucitavanje sirovog genoma
    # BLAST poravnanje

    try:
        translator = ProteinTranslator()

        protein_df, protein_stats = translator.process_mutations_file(
            input_file="data/processed/mutation_with_sequences.csv",
            output_file="data/processed/protein_translations.csv",
            reference_mrna_file="data/raw/hbb_reference.fasta"
        )
    except Exception as e:
        print(f"\n❌ GREŠKA pri translaciji: {e}")
        import traceback
        traceback.print_exc()

        # ========== 6. ONE-HOT ENCODING PROTEINA ==========
    print("\n" + "=" * 70)
    print("\n[KORAK 6] One-Hot Encoding proteina za ML...")

    try:
        encoder = ProteinOneHotEncoder()

        encoded_data = encoder.process_protein_file(
            input_file="data/processed/protein_translations.csv",
            output_file="data/ml_ready/encoded_proteins.npz",
            include_reference=True
        )

        if encoded_data:
            print("\n✓ Enkodovanje završeno!")

            # Pripremi podatke za ML
            print("\n[7] Priprema podataka za Machine Learning...")
            X, y = prepare_for_ml(
                encoded_data,
                flatten=True,
                include_additional=True
            )

            # Train/Test split
            X_train, X_test, y_train, y_test = create_train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )

            print("\n" + "=" * 70)
            print("✓ PODACI SPREMNI ZA MACHINE LEARNING")
            print("=" * 70)
            print(f"\n📊 Dataset info:")
            print(f"   Ukupno uzoraka: {len(y)}")
            print(f"   Features: {X.shape[1]}")
            print(f"   Train set: {X_train.shape[0]} samples")
            print(f"   Test set: {X_test.shape[0]} samples")

            print(f"\n📁 ML-ready fajlovi:")
            print(f"   1. data/ml_ready/encoded_proteins.npz")
            print(f"   2. data/ml_ready/encoded_proteins_metadata.csv")
            print(f"   3. data/ml_ready/encoded_proteins_encoder.pkl")

    except Exception as e:
        print(f"\n❌ GREŠKA pri enkodovanju: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("\n[KORAK 7] Treniranje ML modela za predikciju patogenosti...")

    try:
        classifier, results = train_pathogenicity_model(
            encoded_data_file="data/ml_ready/encoded_proteins.npz",
            output_dir="data/ml_results",
            binary_classification=True  # Pathogenic vs Not Pathogenic
        )

        print("\n✓ Model uspešno istreniran!")
        print(f"\n🏆 Najbolji model: {results['best_model_name']}")
        print(f"   F1-Score: {results['best_score']:.4f}")

    except Exception as e:
        print(f"\n❌ GREŠKA pri treniranju: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
