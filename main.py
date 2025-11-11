import pandas as pd

# df = pd.read_csv("GSE33000_raw_data.txt", sep="\t", comment='!', low_memory=False)
# pd.set_option('display.max_columns', None)  # Show all columns if needed

# print(df['reporterID'])        # Show first 5 rows
# print(df.shape)         # See how many rows × columns
# print(df.columns[:30])

import pandas as pd
import gzip, io, re

def _is_gzip(path):
    with open(path, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def _find_header_index(path):
    # caută primul rând de header care începe cu "ID\t"
    if _is_gzip(path):
        opener = lambda p: io.TextIOWrapper(gzip.open(p, 'rb'), encoding='utf-8', errors='replace')
    else:
        opener = lambda p: open(p, 'r', encoding='utf-8', errors='replace')
    with opener(path) as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith('ID\t'):
                return i
    return None

def read_geo_annot(path):
    header_idx = _find_header_index(path)
    if header_idx is None:
        raise RuntimeError("Nu am găsit linia de header (cea care începe cu 'ID\\t'). Verifică fișierul .annot.")

    compression = 'gzip' if _is_gzip(path) else 'infer'
    # comentariile GEO încep de obicei cu '!' — le ignorăm
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        skiprows=header_idx,   # prima linie după skip devine header
        dtype=str,
        engine="python",
        comment="!",
        compression=compression
        # poți adăuga on_bad_lines="skip" dacă mai rămân rânduri atipice
    )

    # identifică coloanele ID și SYMBOL
    id_candidates  = ["ID"]
    sym_candidates = ["Gene symbol"]

    id_col  = next((c for c in id_candidates  if c in df.columns), None)
    sym_col = next((c for c in sym_candidates if c in df.columns), None)
    if id_col is None or sym_col is None:
        raise ValueError(f"Nu am găsit coloanele pentru ID/SYMBOL. Coloane disponibile: {list(df.columns)}")

    def clean_symbol(x):
        if pd.isna(x) or str(x).strip()=="":
            return None
        return re.split(r"\s*///\s*|;|,", str(x))[0].strip()

    df["_SYMBOL_CLEAN"] = df[sym_col].map(clean_symbol)

    map_id_to_symbol = (
        df.dropna(subset=[id_col, "_SYMBOL_CLEAN"])
          .drop_duplicates(subset=[id_col])
          .set_index(id_col)["_SYMBOL_CLEAN"]
          .to_dict()
    )
    return df, map_id_to_symbol

# utilizare:
path = "GPL4372.annot"  # pune calea ta reală
df_annot, map_id2sym = read_geo_annot(path)

print(map_id2sym)

# exemplu aplicare pe un DataFrame de expresii care are coloana "ID_REF"
# expr["SYMBOL"] = expr["ID_REF"].map(map_id2sym)