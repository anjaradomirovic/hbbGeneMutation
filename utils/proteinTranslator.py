# utils/protein_translator.py

import pandas as pd
from pathlib import Path
from Bio.Seq import Seq
from Bio.Data import CodonTable

#prevod mRNA sekvence u proteine + CSV nova tabela

class ProteinTranslator:

    def __init__(self):

        #Inicijalizacija

        # Standardna genetska kod tabela
        self.codon_table = CodonTable.standard_dna_table

        # Stop kodoni
        self.stop_codons = ['TAA', 'TAG', 'TGA']

        print(f"\n{'=' * 80}")
        print("PROTEIN TRANSLATOR - Inicijalizovan")
        print(f"{'=' * 80}")
        print(f"  Genetski kod: Standardna (NCBI table 1)")
        print(f"  Stop kodoni: {', '.join(self.stop_codons)}")
        print(f"{'=' * 80}\n")

    def translate_mrna(self, mrna_sequence):
        """
        Prevodi mRNA sekvencu u protein sekvencu

        Args:
            mrna_sequence (str): mRNA sekvenca

        Returns:
            dict: {
                'protein_sequence': str,
                'protein_length': int,
                'has_premature_stop': bool,
                'stop_position': int or None,
                'is_complete': bool,
                'error': str or None
            }
        """
        result = {
            'protein_sequence': None,
            'protein_length': 0,
            'has_premature_stop': False,
            'stop_position': None,
            'is_complete': False,
            'error': None
        }

        if not mrna_sequence:
            result['error'] = "Prazna mRNA sekvenca"
            return result

        try:
            # Konvertuj u Bio.Seq objekat
            seq = Seq(mrna_sequence)

            # Proveri duzinu (mora biti deljivo sa 3)
            if len(seq) % 3 != 0:
                # warning
                result['error'] = f"mRNA dužina ({len(seq)}) nije deljiva sa 3 - frameshift"

            # Transliraj
            protein = str(seq.translate(to_stop=False))  # Ne zaustavljaj se na prvom stopu

            # Proveri da li ima preuranjen stop kodon
            stop_positions = []
            for i, aa in enumerate(protein):
                if aa == '*':
                    stop_positions.append(i)

            if stop_positions:
                # Proveri da li je stop kodon na kraju (normalno)
                if stop_positions[-1] == len(protein) - 1:
                    # Normalan stop na kraju
                    result['is_complete'] = True
                    # Ukloni stop kodon sa kraja
                    protein = protein[:-1]

                    # Proveri da li ima i druge stop kodone (preuranjen)
                    if len(stop_positions) > 1:
                        result['has_premature_stop'] = True
                        result['stop_position'] = stop_positions[0]  # Prvi preuranjen stop
                else:
                    # Stop kodon nije na kraju ili ga nema
                    result['has_premature_stop'] = True
                    result['stop_position'] = stop_positions[0]

            result['protein_sequence'] = protein
            result['protein_length'] = len(protein)

            return result

        except Exception as e:
            result['error'] = f"Greška pri translaciji: {str(e)}"
            return result

    def compare_proteins(self, reference_protein, mutated_protein):
        """
        Poredi referentni i mutirani protein

        Returns:
            dict: {
                'is_identical': bool,
                'length_change': int,
                'amino_acid_changes': int,
                'mutations': list of tuples (position, ref_aa, mut_aa),
                'truncated': bool,
                'extended': bool
            }
        """
        comparison = {
            'is_identical': False,
            'length_change': 0,
            'amino_acid_changes': 0,
            'mutations': [],
            'truncated': False,
            'extended': False
        }

        if not reference_protein or not mutated_protein:
            return comparison

        # Dužina
        ref_len = len(reference_protein)
        mut_len = len(mutated_protein)
        comparison['length_change'] = mut_len - ref_len

        # Proveri da li je identičan
        if reference_protein == mutated_protein:
            comparison['is_identical'] = True
            return comparison

        # Proveri truncation/extension
        if mut_len < ref_len:
            comparison['truncated'] = True
        elif mut_len > ref_len:
            comparison['extended'] = True

        # Uporedi aminokiseline
        min_len = min(ref_len, mut_len)
        for i in range(min_len):
            if reference_protein[i] != mutated_protein[i]:
                comparison['mutations'].append((i + 1, reference_protein[i], mutated_protein[i]))
                comparison['amino_acid_changes'] += 1

        return comparison

    def process_mutations_file(self,
                               input_file='data/processed/mutation_with_sequences.csv',
                               output_file='data/processed/protein_translations.csv',
                               reference_mrna_file='data/raw/hbb_reference.fasta'):
        """
        Procesira fajl sa mRNA sekvencama i generiše CSV sa protein translacijama

        Args:
            input_file: CSV sa mutiranim mRNA sekvencama
            output_file: Output CSV sa proteinima
            reference_mrna_file: Referentna mRNA sekvenca
        """
        print(f"\n{'=' * 80}")
        print("PROTEIN TRANSLATION PIPELINE")
        print(f"{'=' * 80}\n")

        # 1. Ucitaj referentnu mRNA
        print(f"1. Učitavanje referentne mRNA iz {reference_mrna_file}...")
        reference_mrna = self._read_fasta(reference_mrna_file)
        print(f"     Referentna mRNA: {len(reference_mrna)} bp")

        # 2. Transliraj referentnu mRNA
        print(f"\n2. Translacija referentne mRNA...")
        ref_translation = self.translate_mrna(reference_mrna)

        if ref_translation['error']:
            print(f"      Warning: {ref_translation['error']}")

        reference_protein = ref_translation['protein_sequence']
        print(f"    Referentni protein: {len(reference_protein)} aminokiselina")
        print(f"    Prvih 20 AA: {reference_protein[:20]}...")

        # 3. Učitaj mutacije
        print(f"\n3. Učitavanje mutacija iz {input_file}...")
        df = pd.read_csv(input_file)
        print(f"     Učitano {len(df)} mutacija")

        # 3.2. Filtriraj samo procesuirane (sa mRNA)
        processed = df[df['processing_status'] == 'PROCESSED'].copy()
        print(f"     3.2 Procesuirano (sa mRNA): {len(processed)} mutacija")

        if len(processed) == 0:
            print("\n      Nema procesuiranih mutacija za translaciju!")
            return None

        # 4. Transliraj sve mutirane mRNA sekvence
        print(f"\n4. Translacija {len(processed)} mutiranih mRNA...")

        translations = []
        stats = {
            'total': len(processed),
            'successful': 0,
            'with_premature_stop': 0,
            'frameshift': 0,
            'identical_protein': 0,
            'protein_changes': 0,
            'truncated': 0,
            'extended': 0,
            'errors': 0
        }

        for idx, row in processed.iterrows():
            variant_id = row.get('variant_id', f'variant_{idx}')
            mutated_mrna = row['mutated_mrna']

            # Transliraj mutiranu mRNA
            translation = self.translate_mrna(mutated_mrna)

            # Uporedi sa referentnim proteinom
            if translation['protein_sequence']:
                comparison = self.compare_proteins(reference_protein, translation['protein_sequence'])

                # Kreiraj rezultat
                translation_result = {
                    'variant_id': variant_id,
                    'original_index': row.get('original_index', idx),

                    # Genomske informacije
                    'chromosome': row.get('chromosome', 'N/A'),
                    'position': row.get('position', 'N/A'),
                    'ref_allele': row.get('ref_allele', 'N/A'),
                    'alt_allele': row.get('alt_allele', 'N/A'),
                    'hgvs': row.get('hgvs', 'N/A'),
                    'clinical_significance': row.get('clinical_significance', 'N/A'),

                    # Lokacija
                    'location_type': row.get('location_type', 'N/A'),
                    'location_detail': row.get('location_detail', 'N/A'),

                    # Splice info
                    'splice_affected': row.get('splice_affected', False),
                    'splice_score': row.get('splice_score', 0.0),
                    'splice_type': row.get('splice_type', 'N/A'),

                    # mRNA
                    'mrna_sequence': mutated_mrna,
                    'mrna_length': len(mutated_mrna),

                    # PROTEIN (referentni)
                    'reference_protein': reference_protein,
                    'reference_protein_length': len(reference_protein),

                    # PROTEIN (mutirani)
                    'mutated_protein': translation['protein_sequence'],
                    'mutated_protein_length': translation['protein_length'],

                    # Translacija info
                    'has_premature_stop': translation['has_premature_stop'],
                    'stop_position': translation['stop_position'],
                    'is_complete': translation['is_complete'],
                    'translation_error': translation['error'],

                    # Poređenje proteina
                    'is_identical_protein': comparison['is_identical'],
                    'protein_length_change': comparison['length_change'],
                    'amino_acid_changes': comparison['amino_acid_changes'],
                    'is_truncated': comparison['truncated'],
                    'is_extended': comparison['extended'],
                    'protein_mutations': self._format_mutations(comparison['mutations']),

                    # Status
                    'translation_status': 'SUCCESS' if not translation['error'] else 'WARNING'
                }

                translations.append(translation_result)

                # Ažuriraj statistiku
                stats['successful'] += 1

                if translation['has_premature_stop']:
                    stats['with_premature_stop'] += 1

                if translation['error'] and 'frameshift' in translation['error']:
                    stats['frameshift'] += 1

                if comparison['is_identical']:
                    stats['identical_protein'] += 1
                else:
                    stats['protein_changes'] += 1

                if comparison['truncated']:
                    stats['truncated'] += 1
                elif comparison['extended']:
                    stats['extended'] += 1

            else:
                stats['errors'] += 1

            # Progress
            if (idx + 1) % 50 == 0:
                print(f"    Translirano: {stats['successful']}/{len(processed)}")

        # 5. Kreiraj DataFrame
        print(f"\n[5] Kreiranje output tabele...")
        result_df = pd.DataFrame(translations)

        # 6. Sacuvaj
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_file, index=False)

        # 7. Statistika
        print(f"\n{'=' * 80}")
        print("STATISTIKA TRANSLACIJE")
        print(f"{'=' * 80}")
        print(f"Ukupno za translaciju: {stats['total']}")
        print(f"\nTranslacija:")
        print(f"  ✓ Uspešno:             {stats['successful']:4d} ({stats['successful'] / stats['total'] * 100:5.1f}%)")
        print(f"  ✗ Greške:              {stats['errors']:4d}")
        print(f"\nProteini:")
        print(
            f"  • Identičan referenci: {stats['identical_protein']:4d} ({stats['identical_protein'] / stats['successful'] * 100:5.1f}%)")
        print(
            f"  • Sa promenama:        {stats['protein_changes']:4d} ({stats['protein_changes'] / stats['successful'] * 100:5.1f}%)")
        print(f"  • Truncated:           {stats['truncated']:4d}")
        print(f"  • Extended:            {stats['extended']:4d}")
        print(f"\nProblemi:")
        print(f"    Preuranjen stop:    {stats['with_premature_stop']:4d}")
        print(f"    Frameshift:         {stats['frameshift']:4d}")
        print(f"\n{'=' * 80}")
        print(f"✓ Protein translacije sačuvane u: {output_file}")
        print(f"{'=' * 80}\n")

        return result_df, stats

    def _read_fasta(self, fasta_file):
        """Čita FASTA fajl"""
        sequence = []
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    continue
                sequence.append(line.upper())
        return ''.join(sequence)

    def _format_mutations(self, mutations):
        """
        Formatira listu mutacija u čitljiv string

        Args:
            mutations: list of tuples (position, ref_aa, mut_aa)

        Returns:
            str: "p.Glu6Val; p.Leu7Pro" ili "None"
        """
        if not mutations:
            return "None"

        # Standardni amino acid kod (1-letter → 3-letter)
        aa_map = {
            'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
            'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
            'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
            'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
            '*': 'Ter'
        }

        formatted = []
        for pos, ref_aa, mut_aa in mutations[:10]:  # Prikaži prvih 10
            ref_3 = aa_map.get(ref_aa, ref_aa)
            mut_3 = aa_map.get(mut_aa, mut_aa)
            formatted.append(f"p.{ref_3}{pos}{mut_3}")

        if len(mutations) > 10:
            formatted.append(f"... (još {len(mutations) - 10})")

        return "; ".join(formatted)