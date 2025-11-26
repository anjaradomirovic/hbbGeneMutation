# utils/sequence_generator.py
import pandas as pd
import re
from Bio import SeqIO

# IZ CSV(HGVS) -> FASTA LISTU
# prevodimo svaki gen posebno i dodajemo ga u listu

class SequenceGenerator:

    def __init__(self):
        """Inicijalizacija generatora."""
        pass

    def load_reference_sequence(self, fasta_file):
        """
        Učitava referentnu sekvencu iz FASTA fajla.

        Args:
            fasta_file: Putanja do FASTA fajla

        Returns:
            str: Referentna sekvenca kao string
        """
        try:
            record = SeqIO.read(fasta_file, "fasta")
            return str(record.seq)
        except Exception as e:
            print(f"Greška pri učitavanju reference: {e}")
            return None

    def parse_hgvs(self, hgvs_notation):
        """
        Parsira HGVS notaciju i ekstraktuje mutaciju.

        Args:
            hgvs_notation: HGVS string, npr. "NM_000518.5:c.20A>T"

        Returns:
            dict: {'position': int, 'ref_base': str, 'alt_base': str} ili None
        """

        hgvs = str(hgvs_notation)

        #debug-provera koja je mutacija
        if ">" in hgvs:
            print("Substitucija")
        elif "del" in hgvs:
            print("Delecija")
        elif "ins" in hgvs:
            print("Insercija")
        elif "_" in hgvs:
            print("Verovatno kompleksna mutacija (npr. del ili ins sa opsegom)")
        else:
            print("Nepoznat format")


        if pd.isna(hgvs_notation):
            return None

        # Parsiranje: c.20A>T
        match = re.search(r'c\.(\d+)([ATCG])>([ATCG])', str(hgvs_notation))

        if match:
            return {
                'position': int(match.group(1)) - 1,  # 0-indexed
                'ref_base': match.group(2),
                'alt_base': match.group(3)
            }

        # Parsiranje deletion: c.20del
        match_del = re.search(r'c\.(\d+)del', str(hgvs_notation))
        if match_del:
            return {
                'position': int(match_del.group(1)) - 1,
                'ref_base': None,
                'alt_base': '',  # Deletion
                'type': 'deletion'
            }

        # Parsiranje insertion: c.20_21insA
        match_ins = re.search(r'c\.(\d+)_(\d+)ins([ATCG]+)', str(hgvs_notation))
        if match_ins:
            return {
                'position': int(match_ins.group(1)) - 1,
                'ref_base': None,
                'alt_base': match_ins.group(3),
                'type': 'insertion'
            }

        return None


    def apply_mutation(self, reference_seq, mutation_info):
        """
        Primenjuje mutaciju na referentnu sekvencu.

        Args:
            reference_seq: Referentna sekvenca (string)
            mutation_info: Rečnik sa info o mutaciji

        Returns:
            str: Mutirana sekvenca ili None
        """
        if mutation_info is None:
            return None

        position = mutation_info['position']
        ref_base = mutation_info['ref_base']
        alt_base = mutation_info['alt_base']
        mutation_type = mutation_info.get('type', 'substitution')

        # Provera validnosti pozicije
        if position < 0 or position >= len(reference_seq):
            return None

        # Substitucija (najčešća)
        if mutation_type == 'substitution':
            # Provera da li se referentna baza podudara
            if ref_base and reference_seq[position].upper() != ref_base.upper():
                print(f"  Upozorenje: Nepodudaranje na poziciji {position + 1}: "
                      f"očekivano {ref_base}, dobijeno {reference_seq[position]}")
                return None

            # Primena mutacije
            mutated_seq = reference_seq[:position] + alt_base + reference_seq[position + 1:]
            return mutated_seq

        # Deletion
        elif mutation_type == 'deletion':
            mutated_seq = reference_seq[:position] + reference_seq[position + 1:]
            return mutated_seq

        # Insertion
        elif mutation_type == 'insertion':
            mutated_seq = reference_seq[:position + 1] + alt_base + reference_seq[position + 1:]
            return mutated_seq

        return None

    def generate_mutated_sequences(self, reference_file, clinvar_csv, output_file):
        """
        Generiše mutirane sekvence za sve mutacije iz ClinVar CSV-a.

        Args:
            reference_file: Putanja do FASTA fajla sa referencom
            clinvar_csv: Putanja do CSV fajla sa ClinVar mutacijama
            output_file: Putanja za izlazni CSV sa mutiranim sekvencama

        Returns:
            DataFrame: Tabela sa generisanim sekvencama
        """
        print("\nGenerisanje mutiranih nukleotidnih lanaca...")

        # Učitaj referentnu sekvencu
        reference_seq = self.load_reference_sequence(reference_file)
        if reference_seq is None:
            print("❌ Greška: Referentna sekvenca nije učitana!")
            return None

        print(f"  ✓ Referentna sekvenca: {len(reference_seq)} bp")

        # Učitaj ClinVar podatke
        try:
            df = pd.read_csv(clinvar_csv)
            print(f"  ✓ Učitano {len(df)} mutacija iz ClinVar")
        except Exception as e:
            print(f"❌ Greška pri učitavanju CSV-a: {e}")
            return None

        # Dodaj nove kolone
        df['reference_sequence'] = None
        df['mutated_sequence'] = None
        df['mutation_position'] = None
        df['ref_nucleotide'] = None
        df['alt_nucleotide'] = None
        df['mutation_type'] = None

        valid_count = 0
        skipped_count = 0

        # Generiši mutiranu sekvencu za svaku mutaciju
        print(f"  Procesiranje mutacija...")

        for idx, row in df.iterrows():
            hgvs = row['hgvs_nucleotide']

            # Parsiranje HGVS
            mutation_info = self.parse_hgvs(hgvs)

            if mutation_info is None:
                skipped_count += 1
                continue

            # Primena mutacije
            mutated_seq = self.apply_mutation(reference_seq, mutation_info)

            if mutated_seq is None:
                skipped_count += 1
                continue

            # Sačuvaj podatke
            df.at[idx, 'reference_sequence'] = reference_seq
            df.at[idx, 'mutated_sequence'] = mutated_seq
            df.at[idx, 'mutation_position'] = mutation_info['position'] + 1  # 1-indexed
            df.at[idx, 'ref_nucleotide'] = mutation_info.get('ref_base', '')
            df.at[idx, 'alt_nucleotide'] = mutation_info.get('alt_base', '')
            df.at[idx, 'mutation_type'] = mutation_info.get('type', 'substitution')

            valid_count += 1

            if valid_count % 50 == 0:
                print(f"    Generisano {valid_count} sekvenci...")

        # Filtriraj samo validne mutacije
        df_valid = df[df['mutated_sequence'].notna()].copy()

        # Sačuvaj u CSV
        try:
            df_valid.to_csv(output_file, index=False)
            print(f"\n✓ Generisano {valid_count} validnih mutiranih sekvenci")
            print(f"  Preskočeno: {skipped_count} mutacija")
            print(f"  Sačuvano u: {output_file}")

            # Statistika
            print(f"\n  Distribucija po tipu mutacije:")
            print(df_valid['mutation_type'].value_counts())

            return df_valid

        except Exception as e:
            print(f"❌ Greška pri čuvanju CSV-a: {e}")
            return None