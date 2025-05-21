from docx import Document          # pip install python-docx
import re, json, pathlib

# ---- configurable bits ----------------------------------------------------
DOCX_PATH  = pathlib.Path("data/neuro7_kb.docx")
JSON_PATH  = pathlib.Path("data/residential_complexes.json")

SECTION_ALIASES = {
    # russian headings that appear in the file ------------------------------
    "1. Общая информация"                : "general_info",
    "2. Цены"                            : "pricing",
    "3. Инфраструктура и преимущества"   : "features",
    "4. Коммерческие условия"            : "financial_conditions",
    "4. Коммерческие условия:"           : "financial_conditions",
    "5. Вопросы о работе менеджеров"     : "managers_info",
    "5. Вопросы о работе менеджеров:"    : "managers_info",
}
# a helper to tolerate trailing colons, extra spaces, etc.
SEC_RE = re.compile(r"^\s*(\d\.)\s*(.+?)(\s*:)?\s*$")
# ---------------------------------------------------------------------------

doc   = Document(DOCX_PATH)
data  = {}
name  = current_key = None                         # track context

for para in doc.paragraphs:
    line = para.text.strip()
    if not line:
        continue                                   # skip blank lines

    # ---------- new residential-complex block ------------------------------
    m = re.match(r"^Название:\s*(.+)$", line, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        data[name] = {k: "" for k in (
            "general_info", "pricing", "features",
            "financial_conditions", "managers_info"
        )}
        current_key = None                         # reset section pointer
        continue

    # ---------- section heading inside that block --------------------------
    if line in SECTION_ALIASES:
        current_key = SECTION_ALIASES[line]
        continue

    # ---------- headings with minor format drift ---------------------------
    m = SEC_RE.match(line)
    if m:
        heading = f"{m.group(1)} {m.group(2)}".strip()
        if heading in SECTION_ALIASES:
            current_key = SECTION_ALIASES[heading]
            continue

    # ---------- regular body paragraph -------------------------------------
    if name and current_key:
        data[name][current_key] += line + "\n"

# --------- tidy up newlines -------------------------------------------------
for comp in data.values():
    for k, txt in comp.items():
        comp[k] = txt.rstrip()

# --------- dump to disk ----------------------------------------------------
with JSON_PATH.open("w", encoding="utf-8") as fp:
    json.dump(data, fp, ensure_ascii=False, indent=2)

print(f"✓ Wrote {len(data)} complexes to {JSON_PATH}")
