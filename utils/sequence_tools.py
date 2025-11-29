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


# PREUZIMANJE REF GENA
# PREUZIMANJE UZORAKA

class SequenceTools:
    def __init__(self, email):
        self.email = email
        Entrez.email = email

    def preuzmi_hbb_referentni(self, output_file = "data/raw/hbb_reference.fasta"):
        refseq_id = "NM_000518.5"
        print(f"Preuzimanje referentne sekvence {refseq_id} sa NCBI")

        try:
            handle = Entrez.efetch(db="nucleotide", id=refseq_id, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "genbank")
            handle.close()

            for feature in record.features:
                if feature.type == "CDS":
                    cds_seq = feature.location.extract(record.seq)

                    cds_record = SeqRecord( cds_seq,
                                            id= refseq_id,
                                            description = "HBB (Hemoglobin Subunit Beta) - Coding Sequence")
                    SeqIO.write(cds_record, output_file, "fasta")

                    print(f"\n✓ Referentna sekvenca uspešno sačuvana u: {output_file}")
                    print(f"  Dužina CDS: {len(cds_seq)} bp")
                    print(f"  Broj aminokiselina: {len(cds_seq) // 3}")

                    return cds_record

            print("CDS region nije pronađen u sekvenci!")
            return None

        except Exception as e:
            print(f"Greška prilikom preuzimanja reference: {e}")
            return None



    def preuzmi_clinvar_hbb_vcf(self, output_file="data/raw/clinvar_hbb.csv"):
        """
        Preuzima HBB mutacije iz ClinVar VCF.
        Automatski briše VCF fajl nakon obrade!
        """

        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"\n✓ CSV fajl već postoji i nije prazan: {output_file}")
            return

            # ClinVar VCF URL
        vcf_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"

        # Kreiraj privremeni fajl koji će se automatski obrisati
        temp_vcf = None

        try:
            print(f"\n[1] Preuzimanje ClinVar VCF fajla")

            # Preuzmi u privremeni fajl
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.vcf.gz') as temp_file:
                temp_vcf = temp_file.name

                response = requests.get(vcf_url, stream=True)
                response.raise_for_status()

                # Prikaži progres
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Svaki 10MB
                        progress = (downloaded / total_size) * 100
                        print(
                            f"    Preuzeto: {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({progress:.1f}%)")

            print(f"Preuzeto u privremeni fajl: {temp_vcf}")
            file_size_mb = os.path.getsize(temp_vcf) / 1024 / 1024
            print(f"Veličina: {file_size_mb:.1f} MB")

            # HBB koordinate
            hbb_chr = "11"
            hbb_start = 5225464
            hbb_end = 5229395

            print(f"\n[2] Filtriranje HBB regiona (chr{hbb_chr}:{hbb_start}-{hbb_end})")

            hbb_mutations = []
            line_count = 0

            # Procesuiraj VCF direktno iz privremenog fajla
            with gzip.open(temp_vcf, 'rt') as f:
                for line in f:
                    line_count += 1

                    if line.startswith('#'):
                        continue

                    parts = line.strip().split('\t')
                    if len(parts) < 8:
                        continue

                    chrom = parts[0]
                    pos = int(parts[1])
                    variant_id = parts[2]
                    ref = parts[3]
                    alt = parts[4]
                    info = parts[7]

                    # Filtriraj HBB region
                    if chrom == hbb_chr and hbb_start <= pos <= hbb_end:
                        clnsig = self._extract_vcf_info(info, 'CLNSIG')
                        clnhgvs = self._extract_vcf_info(info, 'CLNHGVS')
                        geneinfo = self._extract_vcf_info(info, 'GENEINFO')
                        clnvc = self._extract_vcf_info(info, 'CLNVC')
                        clnrevstat = self._extract_vcf_info(info, 'CLNREVSTAT')

                        if geneinfo and 'HBB' in geneinfo:
                            hbb_mutations.append({
                                'variant_id': variant_id,
                                'chromosome': chrom,
                                'position': pos,
                                'ref_allele': ref,
                                'alt_allele': alt,
                                'hgvs': clnhgvs if clnhgvs else 'N/A',
                                'clinical_significance': clnsig if clnsig else 'not_provided',
                                'review_status': clnrevstat if clnrevstat else 'N/A',
                                'variation_type': clnvc if clnvc else 'N/A',
                                'gene': 'HBB'
                            })

                    if line_count % 100000 == 0:
                        print(f"    Obrađeno {line_count:,} linija... (HBB: {len(hbb_mutations)})")

            print(f"    ✓ Ukupno: {line_count:,} linija")
            print(f"    ✓ Pronađeno {len(hbb_mutations)} HBB mutacija")

            # ODMAH OBRIŠI VCF FAJL
            print(f"\n[3] Brisanje privremenog VCF fajla...")
            if os.path.exists(temp_vcf):
                os.remove(temp_vcf)
                print(f"    ✓ Obrisan: {temp_vcf} ({file_size_mb:.1f} MB oslobođeno)")
                temp_vcf = None  # Postavi na None da finally blok ne pokuša ponovo

            # Kreiraj DataFrame
            df = pd.DataFrame(hbb_mutations)

            if df.empty:
                print("\n❌ Nije pronađena nijedna HBB mutacija!")
                return None

            # Mapiraj kodove kliničkog značaja
            clnsig_map = {
                '0': 'Uncertain_significance',
                '1': 'not_provided',
                '2': 'Benign',
                '3': 'Likely_benign',
                '4': 'Likely_pathogenic',
                '5': 'Pathogenic',
                '6': 'drug_response',
                '7': 'association',
                '255': 'other'
            }

            df['clinical_significance'] = df['clinical_significance'].apply(
                lambda x: clnsig_map.get(str(x), x) if pd.notna(x) else 'not_provided'
            )

            # Sačuvaj CSV
            print(f"\n[4] Čuvanje u CSV")
            df.to_csv(output_file, index=False)

            csv_size_kb = os.path.getsize(output_file) / 1024

            print(f"CSV fajl: {output_file}")
            print(f"Veličina CSV-a: {csv_size_kb:.1f} KB")
            print(f"Ukupno mutacija: {len(df)}")
            print(f"VCF fajl je automatski obrisan")

            print(f"\nDISTRIBUCIJA PO KLINIČKOM ZNAČAJU:")
            print(df['clinical_significance'].value_counts())

            print(f"\n🔍 PRIMER (prvih 3 mutacije):")
            print(df[['variant_id', 'position', 'ref_allele', 'alt_allele',
                      'hgvs', 'clinical_significance']].head(3))

            return df

        except Exception as e:
            print(f"\n❌ GREŠKA: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            # Osiguraj da se VCF obriše čak i ako dođe do greške
            if temp_vcf and os.path.exists(temp_vcf):
                try:
                    os.remove(temp_vcf)
                    print(f"\n🗑️  Privremeni VCF fajl obrisan (cleanup)")
                except:
                    pass

    def _extract_vcf_info(self, info_str, key):
        """
        Ekstraktuje vrednost iz VCF INFO polja

        Args:
            info_str: INFO string (npr. "CLNSIG=5;CLNHGVS=NC_000011.10:g.5227002T>C")
            key: Ključ koji tražimo (npr. "CLNSIG")

        Returns:
            Vrednost ili None
        """
        for item in info_str.split(';'):
            if '=' in item:
                k, v = item.split('=', 1)
                if k == key:
                    return v
        return None
