from pathlib import Path

table = r"""
\begin{table}[h]
    \centering
    \begin{tabular}{c|c|l|c}
        Directory & Files & Description & LoC \\ \hline \hline
        {\texttt{ein/}} & \texttt{symbols.py}, \texttt{value.py}, \texttt{term.py} & Abstractions for defining Phi/Yarr & 0 \\ \hline
        \multirow{2}{*}{\texttt{ein/phi/}} & \texttt{phi.py} & Phi grammar and typing rules & 0 \\
        & \texttt{type\_system.py} & Type definitions & 0 \\ \hline
        \multirow{2}{*}{\texttt{ein/frontend/}} & \texttt{layout.py} & Encoding Ein records & 0 \\
        & \texttt{ndarray.py}, \dots & User-facing API, building Phi & 0 \\ \hline
        \multirow{4}{*}{\texttt{ein/midend/}} & \texttt{substitution.py} & Generic substitutions & 0 \\
        & \texttt{lining.py} & Inlining and outlining & 0 \\
        & \texttt{equiv.py}, \texttt{size\_classes.py} & Size equivalence judgements & 0 \\
        & \texttt{structs.py} & Array-of-structs to struct-of-arrays transform & 0 \\ \hline
        \multirow{3}{*}{\texttt{ein/codegen/}} & \texttt{yarr.py} & Yarr grammar and typing rules & 0 \\
        & \texttt{axial.py} & Axials for Yarr & 0 \\
        & \texttt{phi\_to\_yarr.py}, \dots & Compilation scheme & 0 \\ \hline
        \multirow{3}{*}{\texttt{ein/backend/}} & \texttt{naive.py} & Na\"ive Phi interpreter & 0 \\
        & \texttt{array\_backend.py} & Abstract array backend definition & 0 \\
        & \texttt{numpy\_backend.py}, \dots & Execution backends & 0 \\ \hline
        \multirow{2}{*}{\texttt{ein/debug/}} & \texttt{graph.py} & \texttt{graphviz} diagrams for term graphs & 0 \\
        & \texttt{pprint.py} & Printing intermediate Phi and Yarr & 0 \\ \hline
        \texttt{tests/} & \texttt{test\_*.py} & General and feature-specific tests & 0 \\ \hline
        \texttt{tests/suite/} & \makecell{\texttt{deep/*.py}, \texttt{misc/*.py}, \\ \texttt{parboil/*.py}, \texttt{rodinia/*.py}} & Longer \textit{cases} for tests and benchmarks & 0 \\ \hline
        \texttt{tools/} & \texttt{benchmark.py}, \dots & Project tools (e.g. benchmarking) & 0 \\ \hline
        \texttt{.} & \texttt{pyproject.toml}, \dots & Project configuration files & 0
    \end{tabular}
    \caption{Overview of the Ein repository.}
    \label{tab:repository}
\end{table}"""

seen = False
dd = None
cfs = None
n = 0
for line in table.splitlines():
    if line.startswith(" " * 8):
        if not seen:
            seen = True
            print(line)
            continue
        d, fs, _, _ = line.split("&")
        d = d.strip().replace(r"\texttt", "")
        fs = fs.strip().replace(r"\texttt", "")
        # print("d?", d, dd)
        if d:
            dd = d[d.find("{{") + 2 : d.find("}}")]
            cfs = set()
        # print(dd, fs)
        if r"\dots" in fs or "*" in fs:
            ls = (
                Path(f"../{dd}").glob("*")
                if dd != "tests/suite/"
                else Path(f"../{dd}").glob("**/*")
            )
            nfs = [
                f
                for f in map(str, ls)
                if f not in cfs
                and (f.endswith(".py") or f.endswith(".toml") or f.endswith(".yaml"))
            ]
        else:
            nfs = [
                f"../{dd}" + f.replace("\\", "")[1:-1]
                for f in fs.split(", ")
                if f not in cfs
            ]
        for f in nfs:
            cfs.add(f)

        i, j = line.rfind("&") + 2, (
            line.rfind(r"\\") - 1 if r"\\" in line else len(line)
        )
        nn = sum(len(open(f).readlines()) for f in nfs)
        n += nn
        line = line[:i] + str(nn) + line[j:]
    print(line)
print(f"Total: {n}")
