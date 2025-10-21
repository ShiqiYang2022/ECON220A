# utils/latex_table.py
import os
import pandas as pd
import numpy as np

def _format_numeric_4dec(df: pd.DataFrame) -> dict:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return {c: (lambda x: "" if pd.isna(x) else f"{float(x):.4f}") for c in cols}

def _auto_align(df: pd.DataFrame) -> str:
    return "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)

def _apply_booktabs(tabular: str) -> str:
    t = tabular.replace("\\hline", "")
    t = t.replace("\\begin{tabular}", "\\toprule\n\\begin{tabular}")
    t = t.replace("\\end{tabular}", "\\bottomrule\n\\end{tabular}")
    return t.replace("\\\\\n", "\\\\\n\\midrule\n", 1)

def export_df_to_latex(
    df: pd.DataFrame,
    out_tex_path: str,
    caption: str = "Table",
    label: str | None = None,
    index: bool = False,
    align: str | None = None,
    na_rep: str = "",
    escape: bool = False,
    use_booktabs: bool = True,
) -> None:
    formatters = _format_numeric_4dec(df)
    if align is None:
        align = _auto_align(df)
    tabular = df.to_latex(
        index=index,
        escape=escape,
        na_rep=na_rep,
        column_format=align,
        formatters=formatters,
        bold_rows=False,
        longtable=False,
        multicolumn=False,
        multicolumn_format="c",
        caption=None,
        label=None,
        buf=None,
        header=True,
        float_format=None,
        sparsify=False,
    )
    if use_booktabs:
        tabular = _apply_booktabs(tabular)
    if label is None:
        stem = os.path.splitext(os.path.basename(out_tex_path))[0]
        label = "tab:" + stem
    table_env = (
        "\\begin{table}[htbp]\n"
        "    \\centering\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        f"{tabular}\n"
        "\\end{table}\n"
    )
    out_dir = os.path.dirname(out_tex_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_env)

def csv_to_latex(
    csv_path: str,
    out_tex_path: str | None = None,
    caption: str | None = None,
    label: str | None = None,
    index: bool = False,
    align: str | None = None,
    na_rep: str = "",
    escape: bool = False,
    use_booktabs: bool = True,
) -> None:
    df = pd.read_csv(csv_path)
    if out_tex_path is None:
        out_tex_path = os.path.splitext(csv_path)[0] + ".tex"
    if caption is None:
        caption = os.path.basename(os.path.splitext(out_tex_path)[0]).replace("_", " ").title()
    export_df_to_latex(
        df=df,
        out_tex_path=out_tex_path,
        caption=caption,
        label=label,
        index=index,
        align=align,
        na_rep=na_rep,
        escape=escape,
        use_booktabs=use_booktabs,
    )
