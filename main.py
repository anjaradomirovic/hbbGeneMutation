import json

import Bio
import pandas
import pandas as pd
import sklearn
import tensorflow
import matplotlib.pyplot as plt

from utils.HBBMutationProcessor import HBBMutationProcessor
from utils.intron_obrada import HBBStructure
from utils.proteinTranslator import ProteinTranslator
from utils.sequence_tools import SequenceTools
from model.needleman_wunsch import NeedlemanWunsch
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

    # Generisanje mutiranih nukleotidnih lanaca
    print("\n HBBMutationProcessor")

    try:
        # Inicijalizuj procesor
        processor = HBBMutationProcessor(
            structure_file="data/raw/hbb_structure.json",
            fasta_file="data/raw/hbb_reference.fasta"
        )

        # Procesiraj mutacije
        mutations_with_mrna, stats = processor.process_clinvar_file(
            input_file="data/raw/clinvar_hbb.csv",
            output_file="data/processed/mutation_with_sequences.csv"
        )

        print(f"\n✓ Procesuirano mutacija: {stats['processed']}/{stats['total']}")
        print(f"✓ Rezultati sačuvani u: data/processed/mutation_with_sequences.csv")

        # Prikaži primer procesuiranih mutacija
        if stats['processed'] > 0:
            print("\n" + "-" * 80)
            print("PRIMER PROCESUIRANIH MUTACIJA:")
            print("-" * 80)
            processed = mutations_with_mrna[
                mutations_with_mrna['processing_status'] == 'PROCESSED'
                ].head(3)

            for idx, row in processed.iterrows():
                print(f"\n  Mutacija #{idx + 1}:")
                print(f"    Pozicija: {row['position']}")
                print(f"    Ref→Alt: {row['ref_allele']}→{row['alt_allele']}")
                print(f"    Lokacija: {row['location_detail']}")
                mrna = row['mutated_mrna']
                if mrna and len(str(mrna)) > 60:
                    print(f"    mRNA: {str(mrna)[:30]}...{str(mrna)[-30:]}")
                else:
                    print(f"    mRNA: {mrna}")

        # Statistika preskočenih
        if stats['skipped'] > 0:
            print(f"\n⚠ {stats['skipped']} mutacija preskočeno:")
            skipped = mutations_with_mrna[
                mutations_with_mrna['processing_status'] == 'SKIPPED'
                ]
            skip_reasons = skipped['skip_reason'].value_counts()
            for reason, count in skip_reasons.items():
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

    print("\n✓ Proteini uspešno generisani!")

    print("\n Poravnanje sekvenci (BLAST)...")


if __name__ == "__main__":
    main()
