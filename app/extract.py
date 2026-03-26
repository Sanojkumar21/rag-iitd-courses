import pdfplumber
from langchain_core.documents import Document


def table_to_string(table: list) -> str:
    if not table:
        return ""

    headers = [str(c).strip() if c else "" for c in table[0]]
    rows = [" | ".join(headers)]

    for row in table[1:]:
        cells = [str(c).strip() if c else "" for c in row]
        if not any(cells):
            continue
        pairs = []
        for h, c in zip(headers, cells):
            if h and c:
                pairs.append(f"{h}: {c}")
            elif c:
                pairs.append(c)
        rows.append(" | ".join(pairs))

    return "\n".join(rows)


def _in_bbox(obj, bbox) -> bool:
    # check if a char object falls inside a given bounding box
    x, y = obj["x0"], obj["top"]
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _in_any_bbox(obj, bboxes) -> bool:
    return any(_in_bbox(obj, b) for b in bboxes)


def extract_pdf(pdf_path: str) -> list[Document]:
    docs = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            pn = page_num + 1

            tables     = page.extract_tables()
            table_objs = page.find_tables()
            bboxes     = [t.bbox for t in table_objs] if table_objs else []

            # tables first
            for idx, table in enumerate(tables):
                if not table:
                    continue
                tstr = table_to_string(table)
                if not tstr.strip():
                    continue
                docs.append(Document(
                    page_content=f"[TABLE page {pn}]\n{tstr}",
                    metadata={
                        "source":      pdf_path,
                        "page":        pn,
                        "type":        "table",
                        "table_index": idx
                    }
                ))

            # text — skip anything that falls inside a table bbox
            if bboxes:
                text = page.filter(
                    lambda obj: obj.get("object_type") == "char" and not _in_any_bbox(obj, bboxes)
                ).extract_text()
            else:
                text = page.extract_text()

            if text and len(text.strip()) > 50:
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": pdf_path,
                        "page":   pn,
                        "type":   "text"
                    }
                ))

    print(f"extracted {len(docs)} docs  ({pdf_path})")
    return docs