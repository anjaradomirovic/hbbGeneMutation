import gzip
import requests
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import os
import tempfile
import time
import xml.etree.ElementTree as ET
import re

class HBBStructure:
    def __init__(self, email):
        self.email = email
        if email:
            Entrez.email = email

        self.ensembl_server = "https://rest.ensembl.org"
        self.hbb_gene_id = "ENSG00000244734"  # Ensembl Gene ID za HBB

    def preuzmi_iz_ensembl(self):
        """
        Preuzima HBB strukturu sa Ensembl REST API
        """
        print("Intronske i eksonske pozicije")

        try:
            # 1. Preuzmi osnovne informacije o genu
            print(f"\n[1] Preuzimanje gena {self.hbb_gene_id}...")

            response = requests.get(
                f"{self.ensembl_server}/lookup/id/{self.hbb_gene_id}",
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            gene_info = response.json()

            print(f"    Gen: {gene_info['display_name']}")
            print(f"    Hromozom: {gene_info['seq_region_name']}")
            print(f"    Start: {gene_info['start']:,}")
            print(f"    End: {gene_info['end']:,}")
            print(f"    Strand: {gene_info['strand']}")

            # 2. Preuzmi sve transkripte
            print(f"\n[2] Preuzimanje transkripata...")

            response = requests.get(
                f"{self.ensembl_server}/overlap/id/{self.hbb_gene_id}?feature=transcript",
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            transcripts = response.json()

            print(f"    Pronađeno {len(transcripts)} transkripata")

            # 3. Pronađi canonical (glavni) protein-coding transkript
            canonical_transcript = None
            for transcript in transcripts:
                if transcript.get('is_canonical') == 1 and transcript.get('biotype') == 'protein_coding':
                    canonical_transcript = transcript
                    break

            if not canonical_transcript:
                print("    ❌ Nije pronađen protein-coding transkript!")
                return None

            transcript_id = canonical_transcript['id']
            print(f"    ✓ Canonical transkript: {transcript_id}")

            # 4. KLJUČNO: Preuzmi DETALJE transkripta sa expand parametrom
            # Ovo vraća SAMO eksone koji pripadaju ovom transkriptu!
            print(f"\n[3] Preuzimanje eksona za {transcript_id}...")

            response = requests.get(
                f"{self.ensembl_server}/lookup/id/{transcript_id}?expand=1",
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            transcript_data = response.json()

            # Ekstraktuj eksone iz Transcript objekta
            if 'Exon' not in transcript_data:
                print("    ❌ Nema eksona u transkriptu!")
                return None

            exons_data = transcript_data['Exon']
            print(f"    ✓ Pronađeno {len(exons_data)} eksona za canonical transkript")

            # 5. Sortiraj eksone
            # Za minus strand, sortiramo opadajuće po end poziciji
            if gene_info['strand'] == -1:
                exons_sorted = sorted(exons_data, key=lambda x: x['end'], reverse=True)
            else:
                exons_sorted = sorted(exons_data, key=lambda x: x['start'])

            # 6. Ekstraktuj koordinate eksona
            print(f"\n[4] Eksoni:")
            exons = []
            for i, exon in enumerate(exons_sorted, 1):
                exon_info = {
                    'number': i,
                    'id': exon['id'],
                    'start': exon['start'],
                    'end': exon['end'],
                    'length': exon['end'] - exon['start'] + 1
                }
                exons.append(exon_info)
                print(f"    Exon {i}: {exon['start']:,} - {exon['end']:,} ({exon_info['length']} bp)")

            # 7. Izračunaj introne
            print(f"\n[5] Introni:")
            introns = []

            for i in range(len(exons) - 1):
                if gene_info['strand'] == -1:
                    # Minus strand: intron je između end eksona i+1 i start eksona i
                    intron_start = exons[i + 1]['end'] + 1
                    intron_end = exons[i]['start'] - 1
                else:
                    # Plus strand: intron je između end eksona i i start eksona i+1
                    intron_start = exons[i]['end'] + 1
                    intron_end = exons[i + 1]['start'] - 1

                # Proveri da li je intron validan
                if intron_start <= intron_end:
                    intron_length = intron_end - intron_start + 1
                    intron_info = {
                        'number': i + 1,
                        'start': intron_start,
                        'end': intron_end,
                        'length': intron_length
                    }
                    introns.append(intron_info)
                    print(f"    Intron {i + 1}: {intron_start:,} - {intron_end:,} ({intron_length:,} bp)")
                else:
                    print(f"    ⚠️ Intron {i + 1}: Nevažeći intron (start > end)")

            # 8. Preuzmi CDS informacije
            print(f"\n[6] CDS informacija...")

            response = requests.get(
                f"{self.ensembl_server}/sequence/id/{transcript_id}?type=cds",
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            cds_data = response.json()

            cds_length = len(cds_data['seq'])
            print(f"    ✓ CDS dužina: {cds_length} bp ({cds_length // 3} aminokiselina)")

            # 9. Sastavi strukturu
            structure = {
                'gene': {
                    'id': self.hbb_gene_id,
                    'symbol': gene_info['display_name'],
                    'description': gene_info.get('description', 'N/A'),
                    'chromosome': gene_info['seq_region_name'],
                    'start': gene_info['start'],
                    'end': gene_info['end'],
                    'strand': gene_info['strand'],
                    'assembly': gene_info['assembly_name']
                },
                'transcript': {
                    'id': transcript_id,
                    'start': canonical_transcript['start'],
                    'end': canonical_transcript['end'],
                    'biotype': canonical_transcript['biotype']
                },
                'exons': exons,
                'introns': introns,
                'cds': {
                    'length': cds_length,
                    'amino_acids': cds_length // 3
                }
            }

            print(f"\n{'=' * 70}")
            print("✓ STRUKTURA USPEŠNO PREUZETA")
            print(f"{'=' * 70}")
            print(f"Eksoni: {len(exons)}")
            print(f"Introni: {len(introns)}")
            print(f"CDS: {cds_length} bp")

            return structure

        except requests.exceptions.RequestException as e:
            print(f"\n❌ HTTP greška: {e}")
            return None
        except Exception as e:
            print(f"\n❌ Greška: {e}")
            import traceback
            traceback.print_exc()
            return None

    def check_mutation(self, pos):
        exons = []
        introns = []
        for i, (start, end) in enumerate(exons, 1):
            if start <= pos <= end:
                return f"Mutacija je u eksonu {i}"
        for i, (start, end) in enumerate(introns, 1):
            if start <= pos <= end:
                return f"Mutacija je u intronu {i}"
        return "Mutacija nije u HBB genu"
