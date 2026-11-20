"""Write the 3x3 NPIV section into NPIV.tex from npiv/results9*/npiv9_results.json and npiv/seeds/s*/npiv9_results.json.
Usage (from project/): python -I npiv/make_tex9.py NPIV.tex
Rewrites everything between '% BEGIN NPIV9 RESULTS' and '% END NPIV9 RESULTS' (inserted before \\end{document} on first run)."""
import sys, json, glob, re
import numpy as np

TEX = sys.argv[1]
R = {"cross": json.load(open("npiv/results9/npiv9_results.json"))["cross"],
     "panel": json.load(open("npiv/results9p/npiv9_results.json"))["panel"]}
SEEDS = {"cross": [json.load(open(p))["cross"] for p in sorted(glob.glob("npiv/seeds/s*/npiv9_results.json"))],
         "panel": [json.load(open(p))["panel"] for p in sorted(glob.glob("npiv/seedsp/s*/npiv9_results.json"))]}
G = ["kernel", "forest", "net"]; F = ["kernel", "forest", "net"]
NAME = {"kernel": "Gaussian kernel", "forest": "Random forest", "net": "Neural net"}


def f(x, d=2):
    return f"{x:.{d}f}".replace("-", "$-$")


def gof_table():
    L = ["\\begin{center}", "\\small\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llccccccc@{}}",
         " & \\(\\hat p\\) class & \\(R^2\\) & \\(R^2\\) no \\(Z\\) & gain (\\(p\\)) & log dens. & no \\(Z\\) & gain (\\(p\\)) & \\\\", "\\hline"]
    for key, lab in (("cross", "Cross-section (\\(n=101\\))"), ("panel", "Panel (\\(n=1{,}010\\))")):
        L.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{lab}}}}}\\\\")
        for g in G:
            v = R[key]["gof"][g]
            L.append(f" & {NAME[g]} & {f(v['r2'], 3)} & {f(v['r2_noz'], 3)} & {f(v['gain_r2'], 3)} ({f(v['p_r2'], 3)}) & "
                     f"{f(v['ll'], 3)} & {f(v['ll_noz'], 3)} & {f(v['gain_ll'], 3)} ({f(v['p_ll'], 3)}) & \\\\")
        lin = R[key]["lin"]
        L.append(f" & Linear (in-sample) & \\multicolumn{{6}}{{l}}{{\\(F = {f(lin['F'])}\\), partial \\(R^2 = {f(lin['partial_r2'], 3)}\\)}} & \\\\")
    L += ["\\hline", "\\end{tabular}", "\\end{center}"]
    return "\n".join(L)


def results_table(key):
    lin = R[key]["lin"]
    L = ["\\begin{center}", "\\small\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lccc@{}}",
         " & \\(\\hat p\\) = Gaussian kernel & \\(\\hat p\\) = Random forest & \\(\\hat p\\) = Neural net\\\\", "\\hline"]
    for fn in F:
        r1, r2, r3 = [NAME[fn] + " \\(f\\)"], [""], [""]
        for gn in G:
            v = R[key]["combos"][f"{gn}|{fn}"]; q = R[key]["gof"][gn]
            r1.append(f"{f(v['npiv'][1], 2)} ({f(v['se'][1], 2)})"); r2.append(f"[{f(v['ci'][1][0], 2)}, {f(v['ci'][1][1], 2)}]")
            r3.append(f"{{\\scriptsize \\(R^2\\) {f(q['r2'], 3)}, gain {f(q['gain_r2'], 3)} (\\(p\\) {f(q['p_r2'], 2)})}}")
        L += [" & ".join(r1) + "\\\\", " & ".join(r2) + "\\\\", " & ".join(r3) + "\\\\[3pt]"]
    L += ["\\hline", f"\\multicolumn{{4}}{{@{{}}l}}{{Linear 2SLS: {f(lin['beta'])} ({f(lin['se'])})}}\\\\", "\\end{tabular}", "\\end{center}"]
    return "\n".join(L)


def seed_table(key):
    SD = SEEDS[key]
    if not SD:
        return ""
    L = ["\\begin{center}", "\\small\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lccc@{}}",
         " & \\(\\hat p\\) = Gaussian kernel & \\(\\hat p\\) = Random forest & \\(\\hat p\\) = Neural net\\\\", "\\hline"]
    for fn in F:
        row = [NAME[fn] + " \\(f\\)"]
        for gn in G:
            v = [s["combos"][f"{gn}|{fn}"]["npiv"][1] for s in SD] + [R[key]["combos"][f"{gn}|{fn}"]["npiv"][1]]
            row.append(f"{f(min(v))} to {f(max(v))}")
        L.append(" & ".join(row) + "\\\\")
    L += ["\\hline", "\\end{tabular}", "\\end{center}"]
    return "\n".join(L)


SECTION = r"""% BEGIN NPIV9 RESULTS
\section*{Application: nine estimator combinations}
We repeat the exercise for the outcome with the largest 2SLS estimate in Tables 6--9 of Ajzenman, Dominguez, and Undurraga (2023): \texttt{del\_problema3}, the share saying crime is a first or second concern (Table 6, column 1; 2SLS \(= 18.61\), s.e.\ \(6.79\)). We combine three estimator classes for the first stage and three for the structural function, giving nine estimates. Both samples from the previous section are used (\(n=101\) cross-section and \(n=1{,}010\) panel with fixed effects removed).

\subsection*{The first stage \(\hat p(a\mid z,x)\): predicting \(A\) from \((Z, X)\)}
\Cref{thm:iv} needs the whole conditional distribution \(p(a\mid z, x)\), not only the conditional mean. For all three classes we regress \(A\) on \((Z, X)\) and assume \(A = m(Z, X) + \varepsilon\) with \(\varepsilon\) independent of \((Z, X)\). Draws are \(\tilde a_{ij} = \hat m_i + \hat\varepsilon_k + \text{jitter}\), where \(\hat\varepsilon_k\) is resampled from the out-of-fold residuals and the jitter is Gaussian with the Silverman bandwidth; there are \(M = 50\) draws per observation. The mean \(\hat m\) is cross-fitted by 5 folds grouped by municipality, so that no draw uses a model trained on the observation it describes. The assumption of an additive error independent of \((Z, X)\) is ours and is not testable here.
\begin{itemize}
    \item \emph{Gaussian kernel.} Local-constant regression with a Gaussian product kernel and one bandwidth per covariate, chosen by leave-one-out least-squares cross-validation (\texttt{statsmodels} \texttt{KernelReg}, \texttt{cv\_ls}).
    \item \emph{Random forest.} The honest forest described above (half-samples without replacement, a random subset of the covariates at each split, deep trees, leaf values on the other half); the leaf share \(\alpha\) and the number of covariates per split are chosen by grouped 5-fold cross-validation.
    \item \emph{Neural net.} A feed-forward ReLU net trained by gradient descent with a normalized step and momentum (step \(\eta_k=\eta/(1+k/100)\), momentum \(0.5\)). Depth \(\in\{1,2\}\), width \(\in\{8,32\}\), \(\eta\in\{0.02,0.1\}\) and the stopping step (under-optimization as regularization) are chosen by grouped 5-fold cross-validation.
\end{itemize}

\subsection*{The structural function \(f\)}
For each draw set, \(\tilde f\) minimizes the loss of \Cref{thm:iv},
\begin{equation*}
    \frac{1}{n}\sum_{i=1}^n\Big(Y_i - \tfrac{1}{M}\sum_{j=1}^M f(\tilde a_{ij}, x_i)\Big)^2,
\end{equation*}
over one of three classes, with regularization chosen by grouped 5-fold cross-validation of the same loss.
\begin{itemize}
    \item \emph{Gaussian kernel.} A sum of Gaussian bumps centered at the observed \((a, x)\) with a ridge penalty; the bandwidth and ridge are chosen by cross-validation.
    \item \emph{Random forest.} The honest forest with leaf values estimated on the exact loss (as in the previous section).
    \item \emph{Neural net.} A ReLU net trained as above on the exact loss, averaged over 5 initializations for the point estimate (one initialization per bootstrap draw).
\end{itemize}
The effect is the average derivative of \(\tilde f\) in \(a\) (centered difference with step \(0.5\) standard deviations of \(A\), observations between the 5th and 95th percentiles of \(A\)). Standard errors and intervals are from a bootstrap over municipalities (100 draws in the cross-section, 40 in the panel), holding all tuning parameters fixed.

\subsection*{First-stage goodness of fit}
\Cref{tab:fs9} reports, for each \(\hat p\) class, the out-of-fold \(R^2\) of \(\hat m(Z, X)\) and the mean leave-one-out log density of the out-of-fold residual (Gaussian kernel density), each with and without the instrument. The gain is the difference. Permutation \(p\)-values permute \(Z\) across observations, refit with the chosen tuning parameters, and use 50 permutations (so the smallest attainable \(p\) is \(0.02\)). The tuning parameters were chosen on the same folds, so the reported fits are mildly optimistic.
\begin{table}[h]
    \centering
    \caption{First-stage fit by \(\hat p\) class}
    \label{tab:fs9}
""" + gof_table() + r"""
\end{table}

\subsection*{Results}
Each cell is the NPIV average derivative with the bootstrap standard error in parentheses and the 95\% percentile interval in brackets.
\begin{table}[h]
    \centering
    \caption{Cross-section (\(n = 101\)): nine NPIV estimates for \texttt{del\_problema3}}
    \label{tab:cross9}
""" + results_table("cross") + r"""
\end{table}
\begin{table}[h]
    \centering
    \caption{Panel (\(n = 1{,}010\)): nine NPIV estimates for \texttt{del\_problema3}}
    \label{tab:panel9}
""" + results_table("panel") + r"""
\end{table}
""" + r"""
\subsection*{Stability across random seeds}
The estimates depend on the random cross-validation folds and on the draws of \(\tilde a\). \Cref{tab:seed9} gives the minimum and maximum point estimate over the main run and reruns that differ only in the random seed (""" + str(len(SEEDS["cross"])) + r""" reruns for the cross-section, """ + str(len(SEEDS["panel"])) + r""" for the panel).
\begin{table}[h]
    \centering
    \caption{Range of the point estimate over seeds: cross-section (top) and panel (bottom)}
    \label{tab:seed9}
""" + seed_table("cross") + "\n" + seed_table("panel") + r"""
\end{table}

\subsection*{Reading the results}
%%READING%%
% END NPIV9 RESULTS
"""

READING = open("npiv/reading9.tex").read() if len(sys.argv) > 2 or True else ""
SECTION = SECTION.replace("%%READING%%", READING)
s = open(TEX).read()
if "% BEGIN NPIV9 RESULTS" in s:
    s = re.sub(r"% BEGIN NPIV9 RESULTS.*?% END NPIV9 RESULTS\n", lambda m: SECTION, s, flags=re.S)
else:
    s = s.replace("\\end{document}", SECTION + "\n\\end{document}")
open(TEX, "w").write(s)
print("wrote", TEX)
