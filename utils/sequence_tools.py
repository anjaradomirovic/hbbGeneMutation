from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import time
import xml.etree.ElementTree as ET


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

    def preuzmi_clinvar_mutacije(self, output_file="data/raw/clinvar_hbb.csv", max_results=200):
        print(f"\nPreuzimanje HBB mutacija iz ClinVar baze.")

        try:
            # pretraga
            search_term = 'HBB[gene] AND "clinsig pathogenic"[Properties]'

            print(f"  Pretraga: {search_term}")
            handle = Entrez.esearch(db="clinvar", term=search_term, retmax=max_results, usehistory="y")
            search_results = Entrez.read(handle)
            handle.close()


            id_list = search_results["IdList"]
            print("ID-jevi:", id_list)
            print(f"  Pronađeno {len(id_list)} varijanti")

            if not id_list:
                print("  Nije pronađena nijedna varijanta!")
                return None
            # preuzimanje podataka


            mutations_data = []
            print(f"  Preuzimanje podataka o mutacijama...")
            for i, variant_id in enumerate(id_list[:max_results], 1):
                if i % 50 == 0:
                    print(f"Obradjeno {i}/{len(id_list[:max_results])} varijanti.")
                try:
                    # Preuzimanje XML podataka za varijantu
                    handle = Entrez.efetch(db="clinvar", id=variant_id, rettype="vcv", retmode="xml")
                    xml_data = handle.read()
                    handle.close()

                    # Parsiranje sa ElementTree umesto Entrez.read()
                    root = ET.fromstring(xml_data)

                    # Parsiranje podataka
                    mutation_info = self._parse_clinvar_xml(root, variant_id)
                    if mutation_info:
                        mutations_data.append(mutation_info)

                    # Pauza da ne preopteretimo server
                    time.sleep(0.2)

                except ET.ParseError as e:
                    print(f"    XML parse greška kod varijante {variant_id}: {e}")
                    continue
                except Exception as e:
                    print(f"    Greška kod varijante {variant_id}: {e}")
                    continue

            df = pd.DataFrame(mutations_data)

            if not df.empty:
                df.to_csv(output_file, index=False)
                print(f"\n✓ Mutacije uspešno sačuvane u: {output_file}")
                print(f"  Ukupno mutacija: {len(df)}")
                print(f"\n  Distribucija po kliničkom značaju:")
                print(df['clinical_significance'].value_counts())
                return df
            else:
                print("  Nije pronađena nijedna validna mutacija!")
                return None
        except Exception as e:
            print(f"Greška prilikom preuzimanja iz ClinVar: {e}")
            return None

    def _parse_clinvar_xml(self, root, variant_id):
        """
        Parsira ClinVar XML koristeći ElementTree.
        """
        try:
            mutation_info = {
                'variant_id': variant_id,
                'clinical_significance': 'Unknown',
                'hgvs_nucleotide': None,
                'hgvs_protein': None,
                'variation_type': None,
                'review_status': None
            }

            # Pronađi ClassifiedRecord
            classified_record = root.find('.//ClassifiedRecord')
            if classified_record is None:
                return None

            # Ekstrakcija kliničkog značaja
            germline_class = classified_record.find('.//GermlineClassification')
            if germline_class is not None:
                description = germline_class.find('Description')
                if description is not None:
                    mutation_info['clinical_significance'] = description.text

                review_status = germline_class.find('ReviewStatus')
                if review_status is not None:
                    mutation_info['review_status'] = review_status.text

            # Ekstrakcija HGVS notacije
            simple_allele = classified_record.find('.//SimpleAllele')
            if simple_allele is not None:
                # Variation type
                variation_type = simple_allele.get('VariationType')
                if variation_type:
                    mutation_info['variation_type'] = variation_type

                # HGVS expressions
                for hgvs in simple_allele.findall('.//HGVSlist/HGVS'):
                    nuc_expr = hgvs.find('NucleotideExpression')
                    if nuc_expr is not None:
                        expr = nuc_expr.find('Expression')
                        if expr is not None and expr.text:
                            mutation_info['hgvs_nucleotide'] = expr.text

                    prot_expr = hgvs.find('ProteinExpression')
                    if prot_expr is not None:
                        expr = prot_expr.find('Expression')
                        if expr is not None and expr.text:
                            mutation_info['hgvs_protein'] = expr.text

            # Vraćaj samo ako ima validne podatke
            if mutation_info['clinical_significance'] != 'Unknown' and \
                    (mutation_info['hgvs_nucleotide'] or mutation_info['hgvs_protein']):
                return mutation_info

            return None

        except Exception as e:
            return None



