import Bio
import pandas
import sklearn
import tensorflow
import matplotlib.pyplot as plt
from utils.sequence_tools import SequenceTools
from utils.sequence_generator import SequenceGenerator
from model.needleman_wunsch import NeedlemanWunsch
import os


def main():

    # Inicijalizacija alata
    seq_tools = SequenceTools("aradomirovic1824rn@raf.rs")

    # 1. Preuzmi referentnu HBB sekvencu
    reference = seq_tools.preuzmi_hbb_referentni("data/raw/hbb_reference.fasta")

    # 2. Preuzmi mutacije iz ClinVar
    mutations_df = seq_tools.preuzmi_clinvar_mutacije("data/raw/clinvar_hbb.csv", max_results = 200)

    # ===== KORAK 3: Generisanje mutiranih nukleotidnih lanaca =====
    print("\n[KORAK 3/4] Generisanje mutiranih nukleotidnih lanaca iz HGVS...")

    generator = SequenceGenerator()
    mutations_with_seq = generator.generate_mutated_sequences(
        reference_file="data/raw/hbb_reference.fasta",
        clinvar_csv="data/raw/clinvar_hbb.csv",
        output_file="data/raw/mutations_with_sequences.csv"
    )

    if mutations_with_seq is None or len(mutations_with_seq) == 0:
        print("❌ GREŠKA: Nije generisana nijedna mutirana sekvenca!")
        return

    # ===== KORAK 4: Needleman-Wunsch poravnanje =====
    print("\n[KORAK 4/4] Poravnanje sekvenci (Needleman-Wunsch)...")

    aligner = NeedlemanWunsch(
        match_score=1,
        mismatch_penalty=-1,
        gap_penalty=-2
    )

    aligned_df = aligner.align_mutations_from_csv(
        input_csv="data/raw/mutations_with_sequences.csv",
        output_csv="data/processed/aligned_mutations.csv"
    )

    if aligned_df is None:
        print("❌ GREŠKA: Poravnanje nije uspelo!")
        return


    if mutations_df is not None:
        print("\n" + "=" * 60)
        print("Primer prvih 5 mutacija:")
        print("=" * 60)
        print(mutations_df.head())

    # ===== PRIMER PORAVNANJA =====
    if len(aligned_df) > 0:
        print("\n" + "=" * 70)
        print(" " * 20 + "PRIMER PORAVNANJA")
        print("=" * 70)

        first = aligned_df.iloc[0]

        print(f"\n📌 Varijanta ID: {first['variant_id']}")
        print(f"   HGVS notacija: {first['hgvs_nucleotide']}")
        print(f"   Klinički značaj: {first['clinical_significance']}")
        print(f"   Review status: {first['review_status']}")

        print(f"\n🧬 Mutacija:")
        print(f"   Pozicija: {first['mutation_position']}")
        print(f"   Promena: {first['ref_nucleotide']} → {first['alt_nucleotide']}")
        print(f"   Tip: {first['mutation_type']}")

        print(f"\n📈 Rezultati poravnanja:")
        print(f"   Alignment score: {first['alignment_score']}")
        print(f"   Identičnost: {first['identity']:.2%}")
        print(f"   Gaps: {first['gaps']}")
        print(f"   Mismatches: {first['mismatches']}")
        print(f"   Matches: {first['matches']}")

        print(f"\n🔬 Poravnate sekvence (prvih 60 bp):")
        print(f"   Ref: {first['aligned_reference'][:60]}")
        print(f"   Mut: {first['aligned_mutated'][:60]}")

        print("\n" + "=" * 70)
        print("💡 Sledeći korak: Trening neuronske mreže na aligned_mutations.csv")
    print("=" * 70 + "\n")
if __name__ == "__main__":
    main()
