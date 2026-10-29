from dataclasses import dataclass
import pymupdf
import hashlib

BOLD_FLAG = 16

@dataclass
class Textblock:
    text: str
    page: int                                                
    is_bold: bool = False
    bbox: tuple[float, float, float, float] | None = None

@dataclass
class Section:
    title: str
    textblocks: list[Textblock]
    page_start: int
    page_end: int
    section_id: int

    def find_section(self, block: Textblock) -> bool:       
        for textblock in self.textblocks:
            if textblock.text == block.text:
                return True
        return False

@dataclass
class PDF:
    path: str
    sections: list[Section]
    pdf_id: str                               

    @classmethod
    def from_file(cls, path: str) -> "PDF":
        sections = []                                        
        current = None
        
        pdf_id = compute_doc_id(path)

        with pymupdf.open(path) as doc:
            for page in doc:
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if block["type"] != 0:                   
                        continue

                    text = ""
                    bold = False
                    bbox = block["bbox"]

                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"]
                            if span["flags"] & BOLD_FLAG:
                                bold = True
                        text += " "
                    textblock = Textblock(text=text, page=page.number, bbox=bbox, is_bold=bold)

                    
                    if textblock.is_bold:
                        current = Section(title=text.strip(), textblocks=[],
                                          page_start=page.number, page_end=page.number, section_id=len(sections))
                        sections.append(current)
                    else:
                        if current is None:
                            current = Section(title="Preamble", textblocks=[],
                                              page_start=page.number, page_end=page.number, section_id=len(sections))
                            sections.append(current)
                        
                        current.textblocks.append(textblock) # ajout du texte
                        current.page_end = page.number

        return cls(path=path, sections=sections, pdf_id=pdf_id)
    
    
def compute_doc_id(path: str) -> str:
    """Empreinte SHA-256 du contenu du fichier."""
    
    with open(path, "rb") as f:
        h = hashlib.sha256(f.read())
        return h.hexdigest()
        

if __name__ == "__main__":
    pdf = PDF.from_file("Attention_is_all_you_need.pdf")
    print(f"PDF ID: {pdf.pdf_id}")
    for s in pdf.sections:
        print(f"[p.{s.page_start}-{s.page_end}] {s.title}")