import re
from dataclasses import dataclass

import tiktoken

from PDF import PDF, Section

ENCODING_NAME = "cl100k_base"
MAX_TOKENS = 400
OVERLAP_TOKENS = 50
EXCLUDED_SECTIONS = ("references", "bibliography", "acknowledg")


@dataclass
class Chunk:
    """envoyé dans Weaviate. Chaque champ = une propriété."""
    text: str
    pdf_id: str
    section_id: int
    chunk_index: int

    source: str
    section: str
    page_start: int
    page_end: int
    n_tokens: int

    @property
    def chunk_id(self) -> str:
        """Identifiant unique et stable du chunk."""
        return f"{self.pdf_id}:{self.section_id}:{self.chunk_index}"

    def text_for_embedding(self) -> str:
        """Texte à vectoriser : titre de section + texte du chunk."""
        return f"{self.section}\n\n{self.text}"


@dataclass
class Unit:
    """Morceau élémentaire (paragraphe ou phrase) qu'on ne recoupe plus."""
    text: str
    page: int
    n_tokens: int


class Chunker:
    def __init__(
        self,
        encoding_name: str = ENCODING_NAME,
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_pdf(self, pdf: PDF) -> list[Chunk]:
        """Découpe toutes les sections du PDF, sauf les sections exclues."""
        chunks = []
        for section in pdf.sections:
            if self._is_excluded(section):
                continue
            chunks.extend(self.chunk_section(section, pdf))
        return chunks

    def chunk_section(self, section: Section, pdf: PDF) -> list[Chunk]:
        """Découpe une section : unités -> groupes -> chunks."""
        units = self._make_units(section)
        groups = self._group_units(units)
        return [self._build_chunk(group, section, pdf, index) for index, group in enumerate(groups)]

    # --- Étape 1 : textblocks -> unités --------------------------------------
    def _make_units(self, section: Section) -> list[Unit]:
        """Transforme les textblocks en unités qui tiennent toutes sous max_tokens."""
        units = []
        for block in section.textblocks:
            text = clean_text(block.text)
            if not text:
                continue

            n_tokens = self.count_tokens(text)
            if n_tokens <= self.max_tokens:
                units.append(Unit(text=text, page=block.page, n_tokens=n_tokens))
                continue

            # Paragraphe trop long : on le coupe en phrases
            for sentence in split_sentences(text):
                n = self.count_tokens(sentence)
                if n <= self.max_tokens:
                    units.append(Unit(text=sentence, page=block.page, n_tokens=n))
                else:
                    # Phrase encore trop longue : coupe brute
                    for piece in self._hard_split(sentence):
                        units.append(Unit(text=piece, page=block.page, n_tokens=self.count_tokens(piece)))
        return units

    def _hard_split(self, text: str) -> list[str]:
        """Dernier recours : coupe brute tous les max_tokens tokens."""
        tokens = self.encoding.encode(text)
        return [
            self.encoding.decode(tokens[i:i + self.max_tokens])
            for i in range(0, len(tokens), self.max_tokens)
        ]

    # --- Étape 2 : regrouper les unités en chunks ----------------------------
    def _group_units(self, units: list[Unit]) -> list[list[Unit]]:
        """Empile les unités jusqu'à max_tokens, avec chevauchement entre groupes."""
        groups = []
        current = []
        for elt in units:
            if current and sum(u.n_tokens for u in current) + elt.n_tokens > self.max_tokens:
                groups.append(current)
                current = self._overlap_tail(current)
                # Cas piège : chevauchement + nouvelle unité dépassent encore
                if sum(u.n_tokens for u in current) + elt.n_tokens > self.max_tokens:
                    current = []
            current.append(elt)

        if current:
            groups.append(current)
        return groups

    def _overlap_tail(self, units: list[Unit]) -> list[Unit]:
        """Dernières unités du groupe, dans la limite de overlap_tokens."""
        tail = []
        total = 0
        for unit in reversed(units):
            if total + unit.n_tokens <= self.overlap_tokens:
                tail.insert(0, unit)
                total += unit.n_tokens
            else:
                break
        return tail

    # --- Étape 3 : construire l'objet Chunk ----------------------------------
    def _build_chunk(self, group: list[Unit], section: Section, pdf: PDF, index: int) -> Chunk:
        """Assemble le texte d'un groupe et remplit un Chunk."""
        text = " ".join(u.text for u in group)
        return Chunk(
            text=text,
            pdf_id=pdf.pdf_id,
            section_id=section.section_id,
            chunk_index=index,
            source=pdf.path,
            section=section.title,
            page_start=min(u.page for u in group),
            page_end=max(u.page for u in group),
            n_tokens=self.count_tokens(text),
        )

    @staticmethod
    def _is_excluded(section: Section) -> bool:
        """True si le titre contient un des mots de EXCLUDED_SECTIONS."""
        title = section.title.lower()
        return any(word in title for word in EXCLUDED_SECTIONS)


# --- Helpers texte -----------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalise les espaces et recolle les mots coupés en fin de ligne."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text


def split_sentences(text: str) -> list[str]:
    """Découpage naïf en phrases (se trompe sur "et al.", "e.g."...)."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s for s in sentences if s.strip()]


if __name__ == "__main__":
    pdf = PDF.from_file("Attention_is_all_you_need.pdf")
    chunker = Chunker()
    chunks = chunker.chunk_pdf(pdf)

    print(f"{len(chunks)} chunks")
    for c in chunks[:5]:
        print(f"\n[{c.chunk_id}] {c.section} | p.{c.page_start}-{c.page_end} | {c.n_tokens} tokens")
        print(c.text[:200], "...")