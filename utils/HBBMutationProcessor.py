import pandas as pd
import json
from pathlib import Path


class HBBMutationProcessor:
    def __init__(self, structure_file='hbb_structure.json', fasta_file='hbb_reference.fasta'):
        """
        Inicijalizacija procesora sa HBB strukturom i mRNA sekvencom

        Args:
            structure_file: Putanja do JSON fajla sa strukturom gena
            fasta_file: Putanja do FASTA fajla sa referentnom mRNA sekvencom
        """
        # Učitaj strukturu gena
        with open(structure_file, 'r') as f:
            self.structure = json.load(f)

        # Učitaj referentnu mRNA iz FASTA fajla (ako postoji)
        self.mrna_ref = None
        if Path(fasta_file).exists():
            self.mrna_ref = self.read_fasta(fasta_file)
            print(f"✓ Učitana referentna mRNA sekvenca ({len(self.mrna_ref)} bp)")
        else:
            print(f"⚠ UPOZORENJE: {fasta_file} nije pronađen - mRNA neće biti generisana")

        # Izvuci potrebne podatke iz strukture
        self.strand = self.structure['gene']['strand']
        self.gene_start = self.structure['gene']['start']
        self.gene_end = self.structure['gene']['end']
        self.exons = [(e['start'], e['end'], e['number']) for e in self.structure['exons']]
        self.introns = [(i['start'], i['end'], i['number']) for i in self.structure['introns']]

        # Sortiraj eksone po poziciji
        self.exons = sorted(self.exons, key=lambda x: x[0])

        # Stop kodoni (provereno iz literature)
        self.stop_codons = ['TAA', 'TAG', 'TGA']

        print(f"\n{'=' * 80}")
        print(f"Učitana struktura HBB gena:")
        print(f"  Hromozom: {self.structure['gene']['chromosome']}")
        print(f"  Pozicija: {self.gene_start}-{self.gene_end}")
        print(f"  Strand: {self.strand} ({'minus' if self.strand == -1 else 'plus'})")
        print(f"  Eksoni: {len(self.exons)}")
        print(f"  Introni: {len(self.introns)}")
        print(f"{'=' * 80}\n")

    def read_fasta(self, fasta_file):
        """
        Čita FASTA fajl i vraća sekvencu bez header-a
        """
        sequence = []
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Preskoči header linije koje počinju sa >
                if line.startswith('>'):
                    continue
                # Dodaj sekvencu
                sequence.append(line.upper())
        return ''.join(sequence)

    def reverse_complement(self, seq):
        """Vraća reverzni komplement sekvence"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
                      'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
        return ''.join(complement.get(base, base) for base in reversed(seq))

    def is_in_intron(self, position):
        """Proverava da li je pozicija u intronu"""
        for start, end, num in self.introns:
            if start <= position <= end:
                return True, num, (start, end)
        return False, None, None

    def is_in_exon(self, position):
        """Proverava da li je pozicija u eksonu"""
        for start, end, num in self.exons:
            if start <= position <= end:
                return True, num, (start, end)
        return False, None, None

    def is_at_splice_site(self, position):
        """
        Proverava da li je pozicija na splice site (±2bp od granice eksona/introna)
        Ovo su kritične pozicije koje ne smeju da se menjaju
        """
        """ TREBA DA SE NADJE API ZA SPLICEAI i da se proveraava da li utice na spplicing ili ne ako ne 
        i u itronu je mRNA je onda obican
         ako utice trazim da mi kaze gde da se promeni!!"""


        # Provera granica eksona
        for start, end, num in self.exons:
            # Donor site (kraj eksona, početak introna)
            if abs(position - end) <= 2:
                return True, f"donor_site_exon_{num}"
            # Acceptor site (početak eksona, kraj introna)
            if abs(position - start) <= 2:
                return True, f"acceptor_site_exon_{num}"

        return False, None

    def genomic_to_mrna_position(self, genomic_pos):
        """
        Konvertuje genomsku poziciju u mRNA poziciju
        Uzima u obzir minus strand
        """
        mrna_pos = 0

        if self.strand == -1:
            # Za minus strand, eksoni su u obrnutom redosledu u mRNA
            for start, end, num in reversed(self.exons):
                if start <= genomic_pos <= end:
                    # Pozicija je u ovom eksonu
                    # Za minus strand: što je veća genomska pozicija, to je manja mRNA pozicija
                    offset = end - genomic_pos
                    return mrna_pos + offset

                # Dodaj dužinu eksona
                mrna_pos += (end - start + 1)
        else:
            # Za plus strand (ako ikad bude potrebno)
            for start, end, num in self.exons:
                if start <= genomic_pos <= end:
                    offset = genomic_pos - start
                    return mrna_pos + offset
                mrna_pos += (end - start + 1)

        return None  # Pozicija nije u eksonu

    def generate_mutated_mrna(self, position, ref_allel, alt_allel):
        """
        Generiše mutiranu mRNA sekvencu

        Returns:
            (mutated_mrna_str, error_msg)
        """
        if self.mrna_ref is None:
            return None, "Referentna mRNA nije učitana"

        # Konvertuj genomsku poziciju u mRNA poziciju
        mrna_pos = self.genomic_to_mrna_position(position)

        if mrna_pos is None:
            return None, "Pozicija nije u kodujućem regionu"

        # Proveri da li je pozicija validna
        if mrna_pos < 0 or mrna_pos >= len(self.mrna_ref):
            return None, f"mRNA pozicija {mrna_pos} van opsega"

        # Pripremi alele
        ref_allel = ref_allel.upper()
        alt_allel = alt_allel.upper()

        # Za minus strand, moramo reverse complement
        if self.strand == -1:
            ref_allel = self.reverse_complement(ref_allel)
            alt_allel = self.reverse_complement(alt_allel)

        # Proveri da li se ref_allel slaže sa referentnom sekvencom
        if self.mrna_ref[mrna_pos] != ref_allel:
            return None, f"Ref alel {ref_allel} se ne slaže sa mRNA ({self.mrna_ref[mrna_pos]}) na poziciji {mrna_pos}"

        # Kreiraj mutiranu mRNA
        mutated_mrna = list(self.mrna_ref)
        mutated_mrna[mrna_pos] = alt_allel
        mutated_mrna_str = ''.join(mutated_mrna)

        return mutated_mrna_str, None

    def classify_and_process_mutation(self, position, ref_allel, alt_allel):
        """
        Klasifikuje mutaciju i generiše mRNA ako je moguće

        Returns:
            dict sa svim informacijama
        """
        result = {
            'position': position,
            'ref_allel': ref_allel,
            'alt_allel': alt_allel,
            'location_type': None,
            'location_detail': None,
            'can_process': False,
            'skip_reason': None,
            'mutated_mrna': None,
            'mrna_length': None,
            'status': None
        }

        # Provera 1: Da li je na splice site?
        at_splice, splice_type = self.is_at_splice_site(position)
        if at_splice:
            result['location_type'] = 'splice_site'
            result['location_detail'] = splice_type
            result['skip_reason'] = f'Na splice site: {splice_type}'
            result['status'] = 'SKIPPED'
            return result

        # Provera 2: Da li je u intronu?
        in_intron, intron_num, intron_coords = self.is_in_intron(position)
        if in_intron:
            result['location_type'] = 'intron'
            result['location_detail'] = f'intron_{intron_num}'
            result['skip_reason'] = 'U intronu (van kodujućeg regiona)'
            result['status'] = 'SKIPPED'
            return result

        # Provera 3: Da li je u eksonu?
        in_exon, exon_num, exon_coords = self.is_in_exon(position)
        if in_exon:
            result['location_type'] = 'exon'
            result['location_detail'] = f'exon_{exon_num}'
            result['can_process'] = True

            # Pokušaj da generiše mutiranu mRNA
            if self.mrna_ref:
                mutated_mrna, error = self.generate_mutated_mrna(position, ref_allel, alt_allel)

                if error:
                    result['status'] = 'ERROR'
                    result['skip_reason'] = error
                    result['can_process'] = False
                else:
                    result['mutated_mrna'] = mutated_mrna
                    result['mrna_length'] = len(mutated_mrna)
                    result['status'] = 'PROCESSED'
            else:
                result['status'] = 'NO_MRNA_REF'
                result['skip_reason'] = 'Referentna mRNA nije dostupna'
                result['can_process'] = False

            return result

        # Ako nije nigde
        result['location_type'] = 'outside'
        result['skip_reason'] = 'Van granica gena'
        result['status'] = 'SKIPPED'
        return result

    def process_clinvar_file(self, input_file='clinvar_hbb.csv',
                             output_file='mutation_with_sequences.csv'):
        """
        Procesira ClinVar CSV fajl i generiše output sa mRNA sekvencama

        Args:
            input_file: Ulazni ClinVar fajl
            output_file: Izlazni fajl sa sekvencama
        """
        print(f"Učitavam {input_file}...")
        df = pd.read_csv(input_file)
        print(f"✓ Učitano {len(df)} mutacija\n")

        # Lista za rezultate
        results = []

        # Statistika
        stats = {
            'total': len(df),
            'processed': 0,
            'skipped': 0,
            'in_exon': 0,
            'in_intron': 0,
            'at_splice_site': 0,
            'outside': 0,
            'errors': 0
        }

        print("Procesuiranje mutacija...")
        print("=" * 80)

        for idx, row in df.iterrows():
            # Izvuci podatke (prilagodi imena kolona prema svom fajlu)
            position = row.get('position') or row.get('pos') or row.get('Position') or row.get('POS')
            ref_allel = row.get('ref_allele') or row.get('ref') or row.get('Ref') or row.get('REF')
            alt_allel = row.get('alt_allele') or row.get('alt') or row.get('Alt') or row.get('ALT')

            if pd.isna(position) or pd.isna(ref_allel) or pd.isna(alt_allel):
                print(f"⚠ Red {idx + 1}: Nedostaju podaci (pozicija/ref/alt), preskačem")
                continue

            position = int(position)
            ref_allel = str(ref_allel).strip()
            alt_allel = str(alt_allel).strip()

            # Klasifikuj i procesiraj
            classification = self.classify_and_process_mutation(position, ref_allel, alt_allel)

            # Kreiraj rezultat sa svim originalnim kolonama
            result_row = row.to_dict()

            # Dodaj nove kolone
            result_row.update({
                'original_index': idx,
                'location_type': classification['location_type'],
                'location_detail': classification['location_detail'],
                'can_process': classification['can_process'],
                'skip_reason': classification['skip_reason'],
                'mutated_mrna': classification['mutated_mrna'],
                'mrna_length': classification['mrna_length'],
                'processing_status': classification['status']
            })

            # Ažuriraj statistiku
            if classification['location_type'] == 'exon':
                stats['in_exon'] += 1
            elif classification['location_type'] == 'intron':
                stats['in_intron'] += 1
            elif classification['location_type'] == 'splice_site':
                stats['at_splice_site'] += 1
            elif classification['location_type'] == 'outside':
                stats['outside'] += 1

            if classification['status'] == 'PROCESSED':
                stats['processed'] += 1
            elif classification['status'] == 'SKIPPED':
                stats['skipped'] += 1
            elif classification['status'] == 'ERROR':
                stats['errors'] += 1

            results.append(result_row)

            # Prikaz progresa
            if (idx + 1) % 50 == 0 or idx == 0:
                processed_so_far = stats['processed']
                print(f"  Obrađeno: {idx + 1}/{len(df)} | Procesuirano: {processed_so_far}")

        # Kreiraj DataFrame
        result_df = pd.DataFrame(results)

        # Reorganizuj kolone - nove na početku
        new_cols = ['original_index', 'position', 'ref_allel', 'alt_allel',
                    'location_type', 'location_detail', 'can_process',
                    'skip_reason', 'processing_status', 'mutated_mrna', 'mrna_length']

        # Dodaj originalne kolone koje nisu već u new_cols
        original_cols = [col for col in df.columns if col not in new_cols]
        final_cols = new_cols + original_cols

        # Reorder sa samo kolonama koje postoje
        final_cols = [col for col in final_cols if col in result_df.columns]
        result_df = result_df[final_cols]

        # Kreiraj direktorijum ako ne postoji
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sačuvaj
        result_df.to_csv(output_file, index=False)

        # Prikaz statistike
        print("\n" + "=" * 80)
        print("ZAVRŠNA STATISTIKA")
        print("=" * 80)
        print(f"Ukupno mutacija: {stats['total']}")
        print(f"\nLokacije:")
        print(f"  • U eksonu:        {stats['in_exon']:4d} ({stats['in_exon'] / stats['total'] * 100:5.1f}%)")
        print(f"  • U intronu:       {stats['in_intron']:4d} ({stats['in_intron'] / stats['total'] * 100:5.1f}%)")
        print(
            f"  • Na splice site:  {stats['at_splice_site']:4d} ({stats['at_splice_site'] / stats['total'] * 100:5.1f}%)")
        print(f"  • Van gena:        {stats['outside']:4d} ({stats['outside'] / stats['total'] * 100:5.1f}%)")
        print(f"\nStatus procesiranja:")
        print(
            f"  • ✓ Procesuirano (mRNA generisana): {stats['processed']:4d} ({stats['processed'] / stats['total'] * 100:5.1f}%)")
        print(
            f"  • ⊘ Preskočeno:                     {stats['skipped']:4d} ({stats['skipped'] / stats['total'] * 100:5.1f}%)")
        print(
            f"  • ✗ Greške:                         {stats['errors']:4d} ({stats['errors'] / stats['total'] * 100:5.1f}%)")
        print(f"\n{'=' * 80}")
        print(f"✓ Rezultati sačuvani u: {output_file}")
        print(f"{'=' * 80}\n")

        return result_df, stats