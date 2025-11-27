from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
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
                    handle = Entrez.efetch(db="clinvar", id=variant_id, rettype="clinvarset", retmode="xml")
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
                   Parsira ClinVar XML - SA DEBUGGINGOM
                   """
        try:
            mutation_info = {
                'variant_id': variant_id,
                'clinical_significance': None,
                'hgvs_nucleotide': None,
                'hgvs_protein': None,
                'variation_type': None,
                'review_status': None
            }

            # ========== 1. CLINICAL SIGNIFICANCE ==========

            # Pokušaj 1: GermlineClassification (najčešća struktura)
            germline = root.find('.//GermlineClassification/Description')
            if germline is not None and germline.text:
                mutation_info['clinical_significance'] = germline.text

            # Pokušaj 2: Stara struktura
            if not mutation_info['clinical_significance']:
                clin_sig = root.find('.//ClinicalSignificance/Description')
                if clin_sig is not None and clin_sig.text:
                    mutation_info['clinical_significance'] = clin_sig.text

            # Pokušaj 3: RCVList/RCVAccession (još starija struktura)
            if not mutation_info['clinical_significance']:
                for rcv in root.findall('.//RCVList/RCVAccession'):
                    # proveri Interpretation i/ili ClinicalSignificance pod RCV
                    interp = rcv.find('.//Interpretation/Description')
                    if interp is not None and interp.text:
                        mutation_info['clinical_significance'] = interp.text
                        break
                    # fallback: možda je direktno u RCVAccession tagu
                    elif rcv.text:
                        mutation_info['clinical_significance'] = rcv.text
                        break
            # if not mutation_info['clinical_significance']:
            #     rcv = root.find('.//RCVList/RCVAccession/ClassifiedConditionList')
            #     if rcv is not None:
            #         interpretation = root.find('.//Interpretation/Description')
            #         if interpretation is not None and interpretation.text:
            #             mutation_info['clinical_significance'] = interpretation.text

            # Review status
            review = root.find('.//ReviewStatus')
            if review is not None and review.text:
                mutation_info['review_status'] = review.text

            # ========== 2. HGVS NUCLEOTIDE ==========

            hgvs_locations_tried = []

            # Pokušaj 1: SimpleAllele/HGVS
            for hgvs in root.findall('.//SimpleAllele//HGVS'):
                hgvs_locations_tried.append("SimpleAllele")
                nuc_expr = hgvs.find('.//NucleotideExpression/Expression')
                if nuc_expr is not None and nuc_expr.text:
                    mutation_info['hgvs_nucleotide'] = nuc_expr.text

                    prot_expr = hgvs.find('.//ProteinExpression/Expression')
                    if prot_expr is not None and prot_expr.text:
                        mutation_info['hgvs_protein'] = prot_expr.text
                    break

            # Pokušaj 2: MeasureSet
            if not mutation_info['hgvs_nucleotide']:
                for hgvs in root.findall('.//MeasureSet//HGVS'):
                    hgvs_locations_tried.append("MeasureSet")
                    nuc_expr = hgvs.find('.//NucleotideExpression/Expression')
                    if nuc_expr is not None and nuc_expr.text:
                        mutation_info['hgvs_nucleotide'] = nuc_expr.text

                        prot_expr = hgvs.find('.//ProteinExpression/Expression')
                        if prot_expr is not None and prot_expr.text:
                            mutation_info['hgvs_protein'] = prot_expr.text
                        break

            # Pokušaj 3: Measure/AttributeSet (HGVS kao atribut)
            if not mutation_info['hgvs_nucleotide']:
                hgvs_pattern = re.compile(r'^[cgnm]\..*')
                for attr in root.findall('.//Attribute'):
                    attr_type = attr.get('Type', '')
                    hgvs_locations_tried.append(f"Attribute:{attr_type}")

                    # if attr.text:
                    #     # HGVS coding (c.)
                    #     if 'HGVS, coding' in attr_type or 'HGVS, genomic' in attr_type:
                    #         mutation_info['hgvs_nucleotide'] = attr.text
                    #         break
                    if attr.text and hgvs_pattern.match(attr.text):
                        mutation_info['hgvs_nucleotide'] = attr.text
                        break

                    # Prepoznaj po prefiksu
                    elif attr.text.startswith(('c.', 'g.', 'n.', 'm.')) and not attr.text.startswith('p.'):
                            mutation_info['hgvs_nucleotide'] = attr.text
                            break

            # Pokušaj 4: Bilo koji NucleotideExpression
            if not mutation_info['hgvs_nucleotide']:
                hgvs_locations_tried.append("Any NucleotideExpression")
                for nuc_expr in root.findall('.//NucleotideExpression'):
                    expr = nuc_expr.find('Expression')
                    if expr is not None and expr.text:
                        # Proveri da li je za HBB gen
                        sequence_acc = nuc_expr.find('.//SequenceAccession')
                        if sequence_acc is not None:
                            acc = sequence_acc.text
                            # NM_000518 je RefSeq za HBB
                            # if 'NM_000518' in acc or not acc:
                            #     mutation_info['hgvs_nucleotide'] = expr.text
                            #     break
                            if sequence_acc is None or 'NM_000518' in sequence_acc.text:
                                mutation_info['hgvs_nucleotide'] = expr.text
                                break
                        else:
                            mutation_info['hgvs_nucleotide'] = expr.text
                            break

            # Variation type
            simple_allele = root.find('.//SimpleAllele')
            if simple_allele is not None:
                var_type = simple_allele.get('VariationType')
                if var_type:
                    mutation_info['variation_type'] = var_type

            if not mutation_info['variation_type']:
                measure = root.find('.//Measure')
                if measure is not None:
                    var_type = measure.get('Type')
                    if var_type:
                        mutation_info['variation_type'] = var_type

            # ========== DEBUGGING OUTPUT ==========

            has_clin_sig = mutation_info['clinical_significance'] is not None
            has_hgvs = mutation_info['hgvs_nucleotide'] is not None

            if not (has_clin_sig and has_hgvs):
                print(f"\n[SKIP] {variant_id}:")
                print(
                    f"  Clinical Significance: {'✓' if has_clin_sig else '✗'} {mutation_info['clinical_significance']}")
                print(f"  HGVS Nucleotide: {'✓' if has_hgvs else '✗'} {mutation_info['hgvs_nucleotide']}")
                print(f"  HGVS locations tried: {set(hgvs_locations_tried)}")

                # Prikaži XML strukturu za debugging
                if not has_hgvs:
                    print(f"  Available XML tags: {[elem.tag for elem in root.iter()][:20]}")

            # ========== VALIDACIJA ==========

            if has_clin_sig and has_hgvs:
                return mutation_info

            return None

        except Exception as e:
            print(f"[ERROR] {variant_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

