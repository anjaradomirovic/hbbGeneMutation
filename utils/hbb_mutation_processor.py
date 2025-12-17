import pandas as pd
import json
from pathlib import Path
import requests
import time


class HBBMutationProcessor:
    def __init__(self, structure_file='hbb_structure.json', fasta_file='hbb_reference.fasta'):

        #Inicijalizacija procesora sa HBB strukturom i mRNA sekvencom

        # Učitaj strukturu gena
        with open(structure_file, 'r') as f:
            self.structure = json.load(f)

        # Učitaj referentnu mRNA iz FASTA fajla
        self.mrna_ref = None
        if Path(fasta_file).exists():
            self.mrna_ref = self.read_fasta(fasta_file)
            print(f"✓ Učitana referentna mRNA sekvenca ({len(self.mrna_ref)} bp)")
        else:
            print(f"⚠ UPOZORENJE: {fasta_file} nije pronađen")

        # Izvuci potrebne podatke iz strukture
        self.strand = self.structure['gene']['strand']
        self.gene_start = self.structure['gene']['start']
        self.gene_end = self.structure['gene']['end']
        self.chromosome = self.structure['gene']['chromosome']
        self.exons = [(e['start'], e['end'], e['number']) for e in self.structure['exons']]
        self.introns = [(i['start'], i['end'], i['number']) for i in self.structure['introns']]

        # Sortiraj eksone
        self.exons = sorted(self.exons, key=lambda x: x[0])

        # Stop kodoni
        self.stop_codons = ['TAA', 'TAG', 'TGA']

        # SpliceAI threshold (preporučeno: 0.5 za high confidence)
        self.spliceai_threshold = 0.5

        print(f"\n{'=' * 80}")
        print(f"Učitana struktura HBB gena:")
        print(f"  Hromozom: {self.chromosome}")
        print(f"  Pozicija: {self.gene_start}-{self.gene_end}")
        print(f"  Strand: {self.strand} ({'minus' if self.strand == -1 else 'plus'})")
        print(f"  Eksoni: {len(self.exons)}")
        print(f"  Introni: {len(self.introns)}")
        print(f"  SpliceAI threshold: {self.spliceai_threshold}")
        print(f"{'=' * 80}\n")

    def read_fasta(self, fasta_file):
        """Čita FASTA fajl"""
        sequence = []
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    continue
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
        Proverava da li je pozicija na splice site (±2bp)
        """
        for start, end, num in self.exons:
            # Donor site (kraj eksona)
            if abs(position - end) <= 2:
                return True, f"donor_site_exon_{num}"
            # Acceptor site (početak eksona)
            if abs(position - start) <= 2:
                return True, f"acceptor_site_exon_{num}"
        return False, None

    def query_spliceai(self, hgvs_notation=None, position=None, ref_allele=None, alt_allele=None):
        """
        Poziva SpliceAI preko Ensembl VEP REST API
        Koristi HGVS notaciju ako je dostupna, inače konstruiše iz pozicije

        Returns:
            dict: {
                'affects_splicing': bool,
                'delta_score': float,
                'splice_type': str,
                'distance': int,
                'exon_skipping': bool,
                'affected_exons': list,
                'raw_data': dict
            }
        """
        try:
            server = "https://rest.ensembl.org"

            # Koristi HGVS notaciju ako je dostupna
            if hgvs_notation and pd.notna(hgvs_notation) and str(hgvs_notation).strip():
                variant_str = str(hgvs_notation).strip()
                print(f"    [SpliceAI] Proverava variant (HGVS): {variant_str}...")
            elif position and ref_allele and alt_allele:
                # Fallback na chr:pos:ref/alt format
                variant_str = f"{self.chromosome}:{position}:{ref_allele}/{alt_allele}"
                print(f"    [SpliceAI] Proverava variant (koordinate): {variant_str}...")
            else:
                print(f"    [SpliceAI] Greška: Nedostaju parametri za query")
                return self._empty_splice_result()

            # VEP poziv sa SpliceAI pluginom
            ext = f"/vep/human/hgvs/{variant_str}"
            params = {
                "SpliceAI": "1",
                "distance": "500",
                "assembly": "GRCh38"
            }

            response = requests.get(
                f"{server}{ext}",
                headers={"Content-Type": "application/json"},
                params=params,
                timeout=30
            )

            # Rate limiting
            if response.status_code == 429:
                print("    [SpliceAI] Rate limit, čekam 60s...")
                time.sleep(60)
                response = requests.get(
                    f"{server}{ext}",
                    headers={"Content-Type": "application/json"},
                    params=params,
                    timeout=30
                )

            # Provera statusa
            if response.status_code == 400:
                print(f"    [SpliceAI] Bad Request (400) - preskačem variant")
                return self._empty_splice_result()

            response.raise_for_status()
            vep_data = response.json()

            # Parsiraj SpliceAI rezultate
            splice_result = self._parse_spliceai_results(vep_data)

            return splice_result

        except requests.exceptions.RequestException as e:
            print(f"    [SpliceAI] HTTP greška: {e}")
            return self._empty_splice_result()
        except Exception as e:
            print(f"    [SpliceAI] Greška: {e}")
            return self._empty_splice_result()

    def _empty_splice_result(self):
        """Vraća prazan rezultat ako SpliceAI ne radi"""
        return {
            'affects_splicing': False,
            'delta_score': 0.0,
            'splice_type': None,
            'distance': None,
            'exon_skipping': False,
            'affected_exons': [],
            'raw_data': None
        }

    def _parse_spliceai_results(self, vep_data):
        """
        Parsira VEP+SpliceAI rezultate
        """
        splice_result = self._empty_splice_result()

        if not vep_data or len(vep_data) == 0:
            return splice_result

        for variant in vep_data:
            if 'transcript_consequences' not in variant:
                continue

            max_delta = 0.0
            splice_type = None
            affected_exons = []

            for consequence in variant['transcript_consequences']:
                if 'spliceai' in consequence:
                    spliceai = consequence['spliceai']

                    ds_ag = float(spliceai.get('DS_AG', 0))
                    ds_al = float(spliceai.get('DS_AL', 0))
                    ds_dg = float(spliceai.get('DS_DG', 0))
                    ds_dl = float(spliceai.get('DS_DL', 0))

                    scores = {
                        'acceptor_gain': ds_ag,
                        'acceptor_loss': ds_al,
                        'donor_gain': ds_dg,
                        'donor_loss': ds_dl
                    }

                    max_score_type = max(scores, key=scores.get)
                    max_score_value = scores[max_score_type]

                    if max_score_value > max_delta:
                        max_delta = max_score_value
                        splice_type = max_score_type

                    if 'exon' in consequence:
                        affected_exons.append(consequence['exon'])

            if max_delta >= self.spliceai_threshold:
                splice_result['affects_splicing'] = True
                splice_result['delta_score'] = max_delta
                splice_result['splice_type'] = splice_type
                splice_result['affected_exons'] = list(set(affected_exons))

                if 'loss' in splice_type:
                    splice_result['exon_skipping'] = True

        splice_result['raw_data'] = vep_data
        return splice_result

    def generate_spliced_mrna(self, splice_result):
        """
        Generiše mRNA na osnovu SpliceAI predikcije
        """
        if not splice_result['affects_splicing']:
            return self.mrna_ref, "Normalan splicing - nema promena"

        splice_type = splice_result['splice_type']
        affected_exons = splice_result['affected_exons']

        if 'loss' in splice_type and splice_result['exon_skipping']:
            skipped_exons = []
            for exon_num in affected_exons:
                try:
                    exon_num = int(exon_num.split('/')[0])
                    skipped_exons.append(exon_num)
                except:
                    continue

            if skipped_exons:
                new_mrna = self._generate_mrna_without_exons(skipped_exons)
                description = f"Exon skipping: eksoni {skipped_exons} preskočeni"
                return new_mrna, description

        elif 'gain' in splice_type:
            return None, f"Cryptic splice site aktiviran ({splice_type}) - potrebna genomska sekvenca"

        return self.mrna_ref, "Nepoznat efekat splicinga"

    def _generate_mrna_without_exons(self, skipped_exon_numbers):
        """
        Generiše mRNA sekvencu bez spec eksona
        """
        if not self.mrna_ref:
            return None

        mrna_parts = []
        mrna_pos = 0

        for start, end, exon_num in self.exons:
            exon_length = end - start + 1

            if exon_num not in skipped_exon_numbers:
                mrna_parts.append(self.mrna_ref[mrna_pos:mrna_pos + exon_length])

            mrna_pos += exon_length

        return ''.join(mrna_parts)

    def genomic_to_mrna_position(self, genomic_pos):
        """Konvertuje genomsku poziciju u mRNA poziciju"""
        mrna_pos = 0

        if self.strand == -1:
            for start, end, num in reversed(self.exons):
                if start <= genomic_pos <= end:
                    offset = end - genomic_pos
                    return mrna_pos + offset
                mrna_pos += (end - start + 1)
        else:
            for start, end, num in self.exons:
                if start <= genomic_pos <= end:
                    offset = genomic_pos - start
                    return mrna_pos + offset
                mrna_pos += (end - start + 1)

        return None

    def generate_mutated_mrna(self, position, ref_allele, alt_allele):
        """Generiše mutiranu mRNA sekvencu"""
        if self.mrna_ref is None:
            return None, "Referentna mRNA nije učitana"

        mrna_pos = self.genomic_to_mrna_position(position)

        if mrna_pos is None:
            return None, "Pozicija nije u kodujućem regionu"

        if mrna_pos < 0 or mrna_pos >= len(self.mrna_ref):
            return self.mrna_ref, "mRNA pozicija van opsega - vraćam referentnu mRNA"

        ref_allele = ref_allele.upper()
        alt_allele = alt_allele.upper()

        if self.strand == -1:
            ref_allele = self.reverse_complement(ref_allele)
            alt_allele = self.reverse_complement(alt_allele)

        if self.mrna_ref[mrna_pos] != ref_allele:
            ref_allele_rc = self.reverse_complement(ref_allele)
            alt_allele_rc = self.reverse_complement(alt_allele)

            if self.mrna_ref[mrna_pos] == ref_allele_rc:
                print(f"    ⚠ Ref alel neslaganje, koristim reverse complement")
                ref_allele = ref_allele_rc
                alt_allele = alt_allele_rc
            else:
                print(f"    ⚠ Ref alel neslaganje: očekivano {self.mrna_ref[mrna_pos]}, dobijeno {ref_allele}")
                print(f"    → Nastavljam sa mutacijom (force apply alt allele)")

        mutated_mrna = list(self.mrna_ref)
        mutated_mrna[mrna_pos] = alt_allele
        mutated_mrna_str = ''.join(mutated_mrna)

        return mutated_mrna_str, None

    def classify_and_process_mutation(self, position, ref_allele, alt_allele, hgvs=None):
        """
        Klasifikuje mutaciju i generiše mRNA
        """
        result = {
            'position': position,
            'ref_allele': ref_allele,
            'alt_allele': alt_allele,
            'location_type': None,
            'location_detail': None,
            'can_process': False,
            'skip_reason': None,
            'mutated_mrna': None,
            'mrna_length': None,
            'status': None,
            'splice_affected': False,
            'splice_score': 0.0,
            'splice_type': None,
            'splice_description': None
        }

        # Provera 1: Da li je u eksonu?
        in_exon, exon_num, exon_coords = self.is_in_exon(position)
        if in_exon:
            result['location_type'] = 'exon'
            result['location_detail'] = f'exon_{exon_num}'
            result['can_process'] = True

            if self.mrna_ref:
                mutated_mrna, error = self.generate_mutated_mrna(position, ref_allele, alt_allele)

                if error:
                    if "van opsega" in error.lower():
                        result['mutated_mrna'] = self.mrna_ref
                        result['mrna_length'] = len(self.mrna_ref)
                        result['status'] = 'PROCESSED'
                        result['can_process'] = True
                        result['skip_reason'] = error
                    else:
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

        # Provera 2: Da li je na splice site?
        at_splice, splice_site_type = self.is_at_splice_site(position)

        # Provera 3: Da li je u intronu?
        in_intron, intron_num, intron_coords = self.is_in_intron(position)

        # Ako je na splice site ILI u intronu, proveri SpliceAI
        if at_splice or in_intron:
            print(
                f"    Pozicija {position}: {'splice site' if at_splice else f'intron {intron_num}'} - pozivam SpliceAI...")

            splice_result = self.query_spliceai(
                hgvs_notation=hgvs,
                position=position,
                ref_allele=ref_allele,
                alt_allele=alt_allele
            )

            result['splice_affected'] = splice_result['affects_splicing']
            result['splice_score'] = splice_result['delta_score']
            result['splice_type'] = splice_result['splice_type']

            if splice_result['affects_splicing']:
                print(
                    f"    ✓ SpliceAI: Utiče na splicing! (score: {splice_result['delta_score']:.3f}, type: {splice_result['splice_type']})")

                spliced_mrna, splice_desc = self.generate_spliced_mrna(splice_result)

                if spliced_mrna:
                    result['location_type'] = 'splice_affected'
                    result['location_detail'] = splice_site_type if at_splice else f"intron_{intron_num}"
                    result['mutated_mrna'] = spliced_mrna
                    result['mrna_length'] = len(spliced_mrna)
                    result['splice_description'] = splice_desc
                    result['status'] = 'PROCESSED'
                    result['can_process'] = True
                    return result
                else:
                    result['location_type'] = 'splice_affected'
                    result['location_detail'] = splice_site_type if at_splice else f"intron_{intron_num}"
                    result['skip_reason'] = splice_desc
                    result['splice_description'] = splice_desc
                    result['status'] = 'SKIPPED'
                    return result
            else:
                print(f"    ✓ SpliceAI: Ne utiče na splicing (score: {splice_result['delta_score']:.3f})")

                result['location_type'] = 'intron' if in_intron else 'splice_site'
                result['location_detail'] = f'intron_{intron_num}' if in_intron else splice_site_type
                result['mutated_mrna'] = self.mrna_ref
                result['mrna_length'] = len(self.mrna_ref) if self.mrna_ref else None
                result['status'] = 'PROCESSED'
                result['can_process'] = True
                result['skip_reason'] = 'Nema uticaja na splicing - referentna mRNA'
                return result

        # Ako nije nigde
        result['location_type'] = 'outside'
        result['mutated_mrna'] = self.mrna_ref
        result['mrna_length'] = len(self.mrna_ref) if self.mrna_ref else None
        result['status'] = 'PROCESSED'
        result['can_process'] = True
        result['skip_reason'] = 'Van granica gena - referentna mRNA'
        return result

    def process_clinvar_file(self, input_file='clinvar_hbb.csv',
                             output_file='mutation_with_sequences.csv',
                             skip_processed=True):
        """
        Procesira ClinVar CSV fajl - SA PRESKAKANJEM VEĆ OBRAĐENIH

        Args:
            skip_processed: Ako True, preskače redove gde je processing_status == 'PROCESSED'
        """
        print(f"Učitavam {input_file}...")

        # ✅ KLJUČNO: Učitaj OUTPUT fajl ako postoji, inače INPUT
        output_path = Path(output_file)
        if skip_processed and output_path.exists():
            print(f"  → Učitavam postojeći output fajl: {output_file}")
            df = pd.read_csv(output_file)
        else:
            df = pd.read_csv(input_file)

        print(f"✓ Učitano {len(df)} mutacija")

        # Proveri koliko je već obrađeno
        if skip_processed and 'processing_status' in df.columns:
            already_processed = (df['processing_status'] == 'PROCESSED').sum()
            to_process = len(df) - already_processed
            print(f"✓ Već obrađeno: {already_processed} mutacija")
            print(f"→ Preostalo za obradu: {to_process} mutacija\n")
        else:
            print(f"→ Sve mutacije će biti obrađene\n")

        results = []

        stats = {
            'total': len(df),
            'already_processed': 0,
            'newly_processed': 0,
            'skipped': 0,
            'in_exon': 0,
            'in_intron': 0,
            'at_splice_site': 0,
            'splice_affected': 0,
            'outside': 0,
            'errors': 0
        }

        print("Procesuiranje mutacija...")
        print("=" * 80)

        for idx, row in df.iterrows():
            # ✅ KLJUČNA PROVERA: Preskači već obrađene
            if skip_processed:
                # Proveri da li kolone postoje
                has_status = 'processing_status' in df.columns
                has_mrna = 'mutated_mrna' in df.columns

                if has_status and has_mrna:
                    existing_status = str(row.get('processing_status', '')).strip()
                    existing_mrna = row.get('mutated_mrna')

                    # Proveri da li je PROCESSED i ima validnu mRNA
                    is_processed = existing_status == 'PROCESSED'
                    has_valid_mrna = (
                            pd.notna(existing_mrna) and
                            existing_mrna != '' and
                            str(existing_mrna).lower() != 'nan' and
                            len(str(existing_mrna)) > 10  # mRNA mora biti >10 bp
                    )

                    if is_processed and has_valid_mrna:
                        # Već procesiran, samo kopiraj postojeće podatke
                        results.append(row.to_dict())
                        stats['already_processed'] += 1

                        if (idx + 1) % 50 == 0:
                            print(
                                f"  Obrađeno: {idx + 1}/{len(df)} | Već završeno: {stats['already_processed']} | Novo: {stats['newly_processed']}")

                        continue

            # Uzmi potrebne podatke
            position = row.get('position') or row.get('pos')
            ref_allele = row.get('ref_allele') or row.get('ref')
            alt_allele = row.get('alt_allele') or row.get('alt')
            hgvs = row.get('hgvs') or row.get('HGVS') or row.get('hgvs_genomic')

            if pd.isna(position) or pd.isna(ref_allele) or pd.isna(alt_allele):
                results.append(row.to_dict())
                continue

            position = int(position)
            ref_allele = str(ref_allele).strip()
            alt_allele = str(alt_allele).strip()

            print(f"  🔄 Obrađujem red {idx}: {position} {ref_allele}>{alt_allele}")

            # Klasifikuj i procesiraj
            classification = self.classify_and_process_mutation(
                position, ref_allele, alt_allele, hgvs=hgvs
            )

            result_row = row.to_dict()
            result_row.update({
                'original_index': idx,
                'location_type': classification['location_type'],
                'location_detail': classification['location_detail'],
                'can_process': classification['can_process'],
                'skip_reason': classification['skip_reason'],
                'mutated_mrna': classification['mutated_mrna'],
                'mrna_length': classification['mrna_length'],
                'processing_status': classification['status'],
                'splice_affected': classification['splice_affected'],
                'splice_score': classification['splice_score'],
                'splice_type': classification['splice_type'],
                'splice_description': classification['splice_description']
            })

            # Statistika
            if classification['splice_affected']:
                stats['splice_affected'] += 1

            if classification['location_type'] == 'exon':
                stats['in_exon'] += 1
            elif classification['location_type'] == 'intron':
                stats['in_intron'] += 1
            elif classification['location_type'] in ['splice_site', 'splice_affected']:
                stats['at_splice_site'] += 1
            elif classification['location_type'] == 'outside':
                stats['outside'] += 1

            if classification['status'] == 'PROCESSED':
                stats['newly_processed'] += 1
            elif classification['status'] == 'SKIPPED':
                stats['skipped'] += 1
            elif classification['status'] == 'ERROR':
                stats['errors'] += 1

            results.append(result_row)

            if (idx + 1) % 20 == 0:
                print(
                    f"  Obrađeno: {idx + 1}/{len(df)} | Već završeno: {stats['already_processed']} | Novo: {stats['newly_processed']}")

            # Rate limiting za SpliceAI API
            time.sleep(0.3)

        # Sačuvaj rezultate
        result_df = pd.DataFrame(results)

        new_cols = ['original_index', 'position', 'ref_allele', 'alt_allele',
                    'location_type', 'location_detail', 'splice_affected', 'splice_score',
                    'splice_type', 'splice_description', 'can_process',
                    'skip_reason', 'processing_status', 'mutated_mrna', 'mrna_length']

        original_cols = [col for col in df.columns if col not in new_cols]
        final_cols = new_cols + original_cols
        final_cols = [col for col in final_cols if col in result_df.columns]
        result_df = result_df[final_cols]

        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_file, index=False)

        # Finalna statistika
        print("\n" + "=" * 80)
        print("ZAVRŠNA STATISTIKA")
        print("=" * 80)
        print(f"Ukupno mutacija: {stats['total']}")
        print(f"\n🔄 Procesiranje:")
        print(f"  • ✅ Već obrađeno:    {stats['already_processed']:4d} (preskočeno)")
        print(f"  • 🆕 Novo procesuirano: {stats['newly_processed']:4d}")
        print(f"\n📍 Lokacije (novo):")
        print(f"  • U eksonu:        {stats['in_exon']:4d}")
        print(f"  • U intronu:       {stats['in_intron']:4d}")
        print(f"  • Na/utiče splice: {stats['at_splice_site']:4d}")
        print(f"  • Van gena:        {stats['outside']:4d}")
        print(f"\n⚠️  SpliceAI uticaj: {stats['splice_affected']:4d} mutacija utiču na splicing")
        print(f"\n❌ Greške:")
        print(f"  • Preskočeno:      {stats['skipped']:4d}")
        print(f"  • Greške:          {stats['errors']:4d}")
        print(f"\n{'=' * 80}")
        print(f"✓ Rezultati sačuvani u: {output_file}")
        print(f"{'=' * 80}\n")

        return result_df, stats


