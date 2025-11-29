import Bio
import pandas
import pandas as pd
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
    print("\n Generisanje mutiranih nukleotidnih lanaca iz HGVS...")

    generator = SequenceGenerator()
    mutations_with_seq = generator.generate_mutated_sequences(
        reference_file="data/raw/hbb_reference.fasta",
        clinvar_csv="data/raw/clinvar_hbb.csv",
        output_file="data/raw/mutations_with_sequences.csv"
    )

    if mutations_with_seq is None or len(mutations_with_seq) == 0:
        print("GREŠKA: Nije generisana nijedna mutirana sekvenca!")
        return

    # Ucitavanje sirovog genoma
    # BLAST poravnanje
    print("\n Poravnanje sekvenci (BLAST)...")


if __name__ == "__main__":
    main()
