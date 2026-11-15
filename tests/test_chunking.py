"""Tests du parsing et du chunking : aucune connexion externe n'est nécessaire."""

import pytest

from PDF import PDF, Section, Textblock, compute_doc_id
from chunking import Chunker, clean_text, split_sentences


# --- Outils pour construire des documents de test ------------------------------
def make_section(title, paragraphs, section_id=0, page=0):
    blocks = [Textblock(text=p, page=page + i) for i, p in enumerate(paragraphs)]
    return Section(
        title=title,
        textblocks=blocks,
        page_start=page,
        page_end=page + max(len(paragraphs) - 1, 0),
        section_id=section_id,
    )


def make_pdf(sections, pdf_id="abc123", path="doc.pdf"):
    return PDF(path=path, sections=sections, pdf_id=pdf_id)


def paragraph(label, n_words=20):
    """Un paragraphe reconnaissable : « label mot1 mot2 ... »."""
    return f"{label} " + " ".join(f"mot{i}" for i in range(n_words)) + "."


# --- Helpers texte ---------------------------------------------------------------
def test_clean_text_normalise_les_espaces():
    assert clean_text("  deux   espaces\n\net un saut ") == "deux espaces et un saut"


def test_clean_text_recolle_les_mots_coupes_en_fin_de_ligne():
    assert clean_text("trans- former") == "transformer"


def test_split_sentences_coupe_avant_une_majuscule():
    assert split_sentences("Première phrase. Deuxième phrase.") == ["Première phrase.", "Deuxième phrase."]


def test_split_sentences_ne_coupe_pas_avant_une_minuscule():
    # Limite connue du découpage naïf : seule une majuscule déclenche la coupure
    assert split_sentences("Un résultat. de test.") == ["Un résultat. de test."]


# --- Chunker ---------------------------------------------------------------------
def test_section_courte_donne_un_seul_chunk():
    chunker = Chunker()
    pdf = make_pdf([make_section("1 Introduction", ["Un texte très court."])])
    chunks = chunker.chunk_pdf(pdf)
    assert len(chunks) == 1
    assert chunks[0].text == "Un texte très court."
    assert chunks[0].section == "1 Introduction"


def test_section_longue_est_decoupee_et_respecte_la_taille_max():
    chunker = Chunker(max_tokens=60, overlap_tokens=15)
    paragraphs = [paragraph(f"P{i}") for i in range(12)]
    chunks = chunker.chunk_pdf(make_pdf([make_section("2 Méthode", paragraphs)]))
    assert len(chunks) > 1
    assert all(c.n_tokens <= 60 for c in chunks)


def test_chevauchement_entre_deux_chunks_consecutifs():
    chunker = Chunker(max_tokens=60, overlap_tokens=30)
    # Paragraphes courts (≈ 20 tokens) : il en tient 3 par chunk, et 1 tient dans le chevauchement.
    # Un paragraphe plus long que overlap_tokens n'est volontairement pas repris.
    paragraphs = [paragraph(f"P{i}", n_words=8) for i in range(12)]
    chunks = chunker.chunk_pdf(make_pdf([make_section("2 Méthode", paragraphs)]))
    assert len(chunks) > 1
    for before, after in zip(chunks, chunks[1:]):
        # Le dernier paragraphe d'un chunk est repris au début du suivant
        last_paragraph = before.text.split(". ")[-1].rstrip(".")
        assert last_paragraph in after.text


def test_aucun_contenu_perdu_dans_le_decoupage():
    chunker = Chunker(max_tokens=60, overlap_tokens=15)
    paragraphs = [paragraph(f"P{i}") for i in range(12)]
    chunks = chunker.chunk_pdf(make_pdf([make_section("2 Méthode", paragraphs)]))
    all_text = " ".join(c.text for c in chunks)
    for i in range(12):
        assert f"P{i} " in all_text


def test_paragraphe_geant_est_coupe_sans_depasser_la_limite():
    chunker = Chunker(max_tokens=50, overlap_tokens=10)
    # Une seule « phrase » sans ponctuation, bien plus longue que la limite
    texte = " ".join(f"mot{i}" for i in range(400))
    chunks = chunker.chunk_pdf(make_pdf([make_section("3 Résultats", [texte])]))
    assert len(chunks) > 1
    assert all(c.n_tokens <= 50 for c in chunks)


def test_sections_references_et_remerciements_sont_exclues():
    chunker = Chunker()
    pdf = make_pdf([
        make_section("1 Introduction", ["Contenu utile."], section_id=0),
        make_section("References", ["[1] Quelqu'un, 2020."], section_id=1),
        make_section("Acknowledgements", ["Merci à tous."], section_id=2),
    ])
    chunks = chunker.chunk_pdf(pdf)
    assert [c.section for c in chunks] == ["1 Introduction"]


def test_pages_de_depart_et_de_fin_du_chunk():
    chunker = Chunker()
    # trois paragraphes courts, sur les pages 4, 5 et 6 : un seul chunk
    section = make_section("4 Résultats", ["Un.", "Deux.", "Trois."], page=4)
    chunks = chunker.chunk_pdf(make_pdf([section]))
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (4, 6)


def test_identifiant_de_chunk_stable_et_texte_pour_embedding():
    chunker = Chunker()
    pdf = make_pdf([make_section("1 Introduction", ["Un texte."], section_id=3)], pdf_id="deadbeef")
    chunk = chunker.chunk_pdf(pdf)[0]
    assert chunk.chunk_id == "deadbeef:3:0"
    # Le titre de la section est placé avant le texte à vectoriser
    assert chunk.text_for_embedding() == "1 Introduction\n\nUn texte."


def test_chunker_est_deterministe():
    chunker = Chunker(max_tokens=60, overlap_tokens=15)
    pdf = make_pdf([make_section("2 Méthode", [paragraph(f"P{i}") for i in range(12)])])
    ids_1 = [c.chunk_id for c in chunker.chunk_pdf(pdf)]
    ids_2 = [c.chunk_id for c in chunker.chunk_pdf(pdf)]
    assert ids_1 == ids_2


# --- Identifiant du document --------------------------------------------------------
def test_compute_doc_id_depend_du_contenu_pas_du_nom(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "renomme.pdf"
    c = tmp_path / "autre.pdf"
    a.write_bytes(b"meme contenu")
    b.write_bytes(b"meme contenu")
    c.write_bytes(b"contenu different")
    assert compute_doc_id(str(a)) == compute_doc_id(str(b))
    assert compute_doc_id(str(a)) != compute_doc_id(str(c))
    assert len(compute_doc_id(str(a))) == 64  # SHA-256 en hexadécimal
