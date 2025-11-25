# model/needleman_wunsch.py
import pandas as pd


class NeedlemanWunsch:
    """
    Implementacija Needleman-Wunsch algoritma za globalno poravnanje sekvenci.
    Implementirano od nule bez spoljnih biblioteka (osim pandas za CSV).
    """

    def __init__(self, match_score=1, mismatch_penalty=-1, gap_penalty=-2):
        """
        Inicijalizacija sa scoring parametrima.

        Args:
            match_score: Nagrada za poklapanje nukleotida (default: 1)
            mismatch_penalty: Kazna za neslaganje nukleotida (default: -1)
            gap_penalty: Kazna za gap (default: -2)
        """
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def _create_matrix(self, rows, cols, value=0):
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(value)
            matrix.append(row)
        return matrix

    def align(self, seq1, seq2):
        """
        Poravnava dve sekvence koristeći Needleman-Wunsch algoritam.

        Args:
            seq1: Prva sekvenca (referentna)
            seq2: Druga sekvenca (mutirana)

        Returns:
            dict: Rečnik sa rezultatima poravnanja
        """
        n = len(seq1)
        m = len(seq2)

        # Kreiranje matrice
        score_matrix = self._create_matrix(n + 1, m + 1, 0)

        # Prva vrsta i kolona, popunjavanje gap penaltyjem
        for i in range(n + 1):
            score_matrix[i][0] = i * self.gap_penalty
        for j in range(m + 1):
            score_matrix[0][j] = j * self.gap_penalty

        # Popunjavanje score matrice
        # idemo kroz redove spoljna matrica
        for i in range(1, n + 1):
            # idemo kroz kolone unutrasnja matrica
            for j in range(1, m + 1):
                # Score za match/mismatch
                if seq1[i - 1] == seq2[j - 1]:
                    match = score_matrix[i - 1][j - 1] + self.match_score
                else:
                    match = score_matrix[i - 1][j - 1] + self.mismatch_penalty

                # Score za gap u seq2 (deletion)
                delete = score_matrix[i - 1][j] + self.gap_penalty

                # Score za gap u seq1 (insertion)
                insert = score_matrix[i][j - 1] + self.gap_penalty

                # Uzmi maksimalni score
                score_matrix[i][j] = max(match, delete, insert)

        # Traceback - rekonstrukcija poravnanja
        aligned_seq1 = ""
        aligned_seq2 = ""
        i = n
        j = m

        while i > 0 or j > 0:
            current_score = score_matrix[i][j]

            # Dijagonala (match ili mismatch)
            if i > 0 and j > 0:
                if seq1[i - 1] == seq2[j - 1]:
                    diagonal_score = score_matrix[i - 1][j - 1] + self.match_score
                else:
                    diagonal_score = score_matrix[i - 1][j - 1] + self.mismatch_penalty

                if current_score == diagonal_score:
                    aligned_seq1 = seq1[i - 1] + aligned_seq1
                    aligned_seq2 = seq2[j - 1] + aligned_seq2
                    i -= 1
                    j -= 1
                    continue

            # Gap u seq2 (deletion iz seq1)
            if i > 0 and current_score == score_matrix[i - 1][j] + self.gap_penalty:
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            # Gap u seq1 (insertion u seq2)
            elif j > 0 and current_score == score_matrix[i][j - 1] + self.gap_penalty:
                aligned_seq1 = "-" + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                j -= 1
            else:
                # Ne bi trebalo da se desi
                break

        # Računanje statistika
        alignment_score = score_matrix[n][m]
        gaps = aligned_seq1.count('-') + aligned_seq2.count('-')

        # Mismatches (neslaganja)
        mismatches = 0
        for a, b in zip(aligned_seq1, aligned_seq2):
            if a != b and a != '-' and b != '-':
                mismatches += 1

        # Matches (poklapanja)
        matches = 0
        for a, b in zip(aligned_seq1, aligned_seq2):
            if a == b:
                matches += 1

        # Identičnost
        identity = matches / len(aligned_seq1) if len(aligned_seq1) > 0 else 0

        return {
            'aligned_seq1': aligned_seq1,
            'aligned_seq2': aligned_seq2,
            'alignment_score': alignment_score,
            'gaps': gaps,
            'mismatches': mismatches,
            'matches': matches,
            'identity': identity
        }

