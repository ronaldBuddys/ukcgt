
import re
import numpy as np
import pandas as pd
from pylatex.utils import NoEscape


def dataframe_to_tex(df, date_format = "%Y-%m-%d", float_format='%.2f'):
    df = df.copy(True)

    # numbers right align, string left align
    column_format = ""
    for idx, c in enumerate(df.columns):
        _ = df[c].dtype
        column_format += 'r' if re.search("^int|^float", str(_)) else 'l'

        # check if the previous column (first level) is different
        if isinstance(c, tuple):
            try:
                if c[0] != df.columns[idx + 1][0]:
                    column_format += "|"
            except:
                pass

    # sanitize string columns
    for c in df.columns:
        # this can be handled by escape=True (?)
        if str(df[c].dtype) == "object":
            pass
            # df[c] = [sanitize_for_latex(_) for _ in df[c]]
        elif re.search("^datetime", str(df[c].dtype)):
            df[c] = df[c].dt.strftime(date_format)
        # TODO: is this needed? why not just use float_format?
        elif re.search("^float", str(df[c].dtype)):
            df[c] = ['na' if np.isnan(j) else float_format % j for j in df[c]]

    out = df.to_latex(index=False, column_format=column_format, escape=True,
                      longtable=True)

    # HACK: hardcoded - to get top columns placed in the center
    out = out.replace('\\multicolumn{3}{r}', '\\multicolumn{3}{c}')

    # # add hline before line that starts with TOTAL
    lines = out.split('\n')
    new_lines = []
    for line in lines:
        if re.search("^TOTAL", line):
            new_lines.append("\\hline")
        new_lines.append(line)

    # Join the lines back into a single string
    out = '\n'.join(new_lines)

    return out


def dict_to_doc(doc, res, new_page=False, current_section="section"):
    section_rank = ["section", "subsection", "subsubsection", "paragraph"]

    assert current_section in section_rank, f"{current_section} not in {section_rank}"

    next_idx = section_rank.index(current_section) + 1
    if next_idx >= len(section_rank):
        next_idx -= 1
    next_section = section_rank[next_idx]

    for k, v in res.items():

        if new_page:
            doc.append(NoEscape('\\newpage'))

        doc.append(NoEscape('\\%s{%s}' % (current_section, k)))

        if isinstance(v, pd.DataFrame):
            doc.append(NoEscape(dataframe_to_tex(v)))
        elif isinstance(v, str):
            doc.append(NoEscape(v))

        elif isinstance(v, list):

            for vidx, vv in enumerate(v):
                if isinstance(vv, pd.DataFrame):
                    doc.append(NoEscape(dataframe_to_tex(vv)))
                elif isinstance(vv, str):
                    doc.append(NoEscape(vv))
                else:
                    raise NotImplementedError(f"list (idx:{vidx}) contained type: {type(vv)}, not handled")

        elif isinstance(v, dict):
            dict_to_doc(doc, v, new_page=False, current_section=next_section)
        else:
            raise NotImplementedError(f"{k} - type:  {type(v)} not implemented")


def latex_escape_ccy(net):
    # is this needed?
    # HACK: escape \&
    # TODO: move this into run()
    all_ccy = np.unique(np.concatenate([net['buy_ccy'].unique(), net['sell_ccy'].unique()]))
    ccy_rename = {c: re.sub("&", "\\&", c) for c in all_ccy}
    net['buy_ccy'] = net['buy_ccy'].map(ccy_rename)
    net['sell_ccy'] = net['sell_ccy'].map(ccy_rename)

    return net


if __name__ == "__main__":

    pass
