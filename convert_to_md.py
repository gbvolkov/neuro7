# pip install mammoth  (run this once from your shell)

import mammoth
from pathlib import Path

def docx_to_markdown(docx_path: str | Path, md_path: str | Path | None = None) -> str:
    """
    Convert a .docx file to Markdown.
    
    Parameters
    ----------
    docx_path : Path or str
        Path to the source .docx file.
    md_path : Path or str | None
        Where to write the Markdown.  
        If None, nothing is written; the Markdown string is just returned.

    Returns
    -------
    str
        The generated Markdown text.
    """
    docx_path = Path(docx_path)
    md_path = Path(md_path) if md_path else None

    with docx_path.open("rb") as docx_file:
        result = mammoth.convert_to_markdown(docx_file)
        markdown: str = result.value                # The Markdown
        messages   = result.messages                # Conversion warnings, if any

    if md_path:
        md_path.write_text(markdown, encoding="utf-8")

    # Optional: surface conversion messages (useful for debugging)
    for m in messages:
        print(f"[{m.type}] {m.message}")

    return markdown


# ----- usage -----
if __name__ == "__main__":
    import os
    for fname in os.listdir("data/"):
        name, ext = os.path.splitext(fname)
        if ext == ".docx":
            md = docx_to_markdown(f"data/{fname}", f"data/{name}.md")
    print("Converted to Markdown:")
    print(md[:500], "…")   # preview first 500 characters