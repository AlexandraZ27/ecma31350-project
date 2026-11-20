"""Write the NPIV application section into NPIV.tex from npiv/results/npiv_results.json.
Usage (from project/): python -I npiv/make_tex.py npiv/results/npiv_results.json NPIV.tex
The section lives between '% BEGIN NPIV RESULTS' and '% END NPIV RESULTS' and is rewritten on each run."""
import sys, json, re

R = json.load(open(sys.argv[1])); TEX = sys.argv[2]
LAB = {"pc_19": "Concerns: PCI", "pc_20": "Preventive behavior: PCI",
       "del_tend_barrio": "Crime rising: neighborhood", "del_tend_com": "Crime rising: municipality",
       "del_tend_pais": "Crime rising: country", "vict_agreg": "Victimization: total"}


def f(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$")


def first_stage_table():
    c, p = R["cross"]["first_stage"], R["panel"]["first_stage"]
    row = lambda lab, g: f"{lab} & {g(c)} & {g(p)}\\\\"
    L = ["\\begin{center}", "\\begin{tabular}{@{}lcc@{}}", " & Cross-section & Panel\\\\", "\\hline",
         row("Observations", lambda s: f"{int(s['n']):,}".replace(",", "{,}")),
         "\\multicolumn{3}{@{}l}{\\emph{Kernel first stage, leave-one-out}}\\\\",
         row("\\(R^2\\) of \\(\\E(A\\mid Z, X)\\), with \\(Z\\)", lambda s: f(s["loo_r2"], 3)),
         row("\\(R^2\\), without \\(Z\\)", lambda s: f(s["loo_r2_noz"], 3)),
         row("Gain (permutation \\(p\\))", lambda s: f"{f(s['gain_r2'], 3)} ({f(s['perm_p_r2'], 3)})"),
         row("Mean log density of \\(A\\), with \\(Z\\)", lambda s: f(s["loo_ll_mean"], 3)),
         row("Mean log density, without \\(Z\\)", lambda s: f(s["loo_ll_mean_noz"], 3)),
         row("Gain (permutation \\(p\\))", lambda s: f"{f(s['gain_ll_mean'], 3)} ({f(s['perm_p_ll'], 3)})"),
         row("Bandwidths \\(h_z / h_x / h_a\\) (sd units)", lambda s: f"{f(s['h_z'])} / {f(s['h_x'], 1)} / {f(s['h_a'])}"),
         "\\multicolumn{3}{@{}l}{\\emph{Linear first stage, in-sample}}\\\\",
         row("Coefficient on instrument (s.e.)", lambda s: f"{f(s['linear']['pi'])} ({f(s['linear']['se_pi'])})"),
         row("\\(F\\)-statistic", lambda s: f(s["linear"]["F"])),
         row("Partial \\(R^2\\)", lambda s: f(s["linear"]["partial_r2"], 3)),
         "\\hline", "\\end{tabular}", "\\end{center}"]
    return "\n".join(L)


def results_table(key):
    L = ["\\begin{center}", "\\small\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lccccc@{}}",
         "Outcome & 2SLS (s.e.) & NPIV (s.e.) & NPIV 95\\% CI & \\(\\alpha\\) & ridge\\\\", "\\hline"]
    for y, lab in LAB.items():
        v = R[key]["outcomes"][y]; k = 1                       # headline step: 0.5 sd
        L.append(f"{lab} & {f(v['lin']['beta'])} ({f(v['lin']['se'])}) & {f(v['npiv'][k], 3)} ({f(v['se'][k], 3)}) & "
                 f"[{f(v['ci'][k][0], 3)}, {f(v['ci'][k][1], 3)}] & {v['alpha']:g} & {v['ridge']:g}\\\\")
    L += ["\\hline", "\\end{tabular}", "\\end{center}"]
    return "\n".join(L)


def step_range(key):
    v = [R[key]["outcomes"][y]["npiv"] for y in LAB]
    return max(max(x) - min(x) for x in v)


SECTION = r"""% BEGIN NPIV RESULTS
\section*{Application: Ajzenman, Dominguez, and Undurraga (2023)}
We apply the two-step procedure of \Cref{thm:iv} to the instrumental-variable specifications of Ajzenman, Dominguez, and Undurraga (AEJ: Applied, 2023), ``Immigration, Crime, and Crime (Mis)Perceptions.'' The treatment \(A\) is immigration in a Chilean municipality, the instrument \(Z\) is the shift-share (Bartik) predicted change in immigration built from migration flows to countries other than Chile, \(X\) is age and gender, and \(Y\) is a municipality-level outcome on a 0--100 scale. We run the six summary outcomes: the principal-component indices of crime-related concerns and of preventive behavior, the three beliefs about crime trends, and total victimization.

\subsection*{Two samples}
\begin{enumerate}
    \item \vocab{Cross-section} (Tables 6--9 of the paper). One observation per municipality (\(n = 101\)): \(A = \Delta\mathrm{migr}\), the 2008--2017 change in the log immigrant-to-population ratio; \(Z\) the shift-share instrument for that change; \(Y\) the 2017--2008 change in the municipality mean of the outcome; \(X\) the municipality mean age and gender share.
    \item \vocab{Panel} (Table A.4, Panel B, collapsed to municipality-year means; \(n = 1{,}010\)). \(A = \log\) immigration rate, \(Z\) the year-specific instrument, \(Y\) and \(X\) the municipality-year means. Municipality and year fixed effects are removed by two-way demeaning of \(Y, A, Z, X\) before estimation, which assumes they enter additively: \(Y = f(A, X) + \delta_m + \lambda_t + \text{error}\).
\end{enumerate}
The estimand in both samples is the average derivative \(\frac{1}{n}\sum_i \partial_a \tilde f(A_i, X_i)\), comparable to the linear 2SLS coefficient.

\subsection*{Implementation}
Tuning follows the lecture notes.
\begin{enumerate}
    \item \emph{Fit \(\hat p(a\mid x, z)\).} A Gaussian conditional kernel density with a product kernel and one bandwidth each for \(z\), \(x\), and \(a\) (all variables standardized). The bandwidths \((h_z, h_x)\) minimize the leave-one-out cross-validation squared error of the kernel conditional mean \(\E(A\mid Z, X)\), so they maximize first-stage fit; \(h_a\) then maximizes the leave-one-out density likelihood.
    \item \emph{Fit \(\tilde f\).} For each observation we draw \(M = 50\) values \(\tilde a_{ij}\sim\hat p(\cdot\mid z_i, x_i)\). \(\tilde f\) is an \vocab{honest random forest} of \(B = 500\) trees. Each tree uses a subsample of size \(n/2\) drawn without replacement and split into two halves \(I_1, I_2\). A deep tree is grown on \(I_1\) (a random 2 of the 3 covariates considered at each split, each leaf containing at least a share \(\alpha\) of \(I_1\)). The leaf values are estimated on \(I_2\) by minimizing the loss in \Cref{thm:iv} itself,
    \begin{equation*}
        \sum_{i\in I_2}\Big(Y_i - \tfrac{1}{M}\sum_{j=1}^M f(\tilde a_{ij}, x_i)\Big)^2 + \text{ridge}.
    \end{equation*}
    The share \(\alpha\in\{0.02, 0.05, 0.10, 0.20\}\) and the ridge penalty are chosen by 5-fold cross-validation of this loss, with folds grouped by municipality.
    \item \emph{Effect.} Centered finite differences of \(\tilde f\) in \(a\), with step \(0.5\) standard deviations of \(A\), averaged over observations between the 5th and 95th percentiles of \(A\). Standard errors and 95\% intervals are from a bootstrap over municipalities (200 draws, 100 trees each), which holds the bandwidths, \(\alpha\), and ridge fixed at their full-sample values.
\end{enumerate}
\answer{\textbf{Why the exact loss.} A simpler alternative regresses \(Y_i\) on the draws \((\tilde a_{ij}, x_i)\) directly. That minimizes \(\E[(Y - f(\tilde A))^2]\), which equals the loss above plus \(\operatorname{Var}(f(\tilde A)\mid z, x)\), and it attenuates the slope toward zero when the instrument is weak. In a simulation with known slope 10 and a weak instrument it returned 1.8, so we do not use it.}

\subsection*{First-stage goodness of fit}
\Cref{tab:fs} reports how well the kernel first stage predicts the treatment out of sample, with and without the instrument. The permutation \(p\)-values permute \(Z\) across observations (200 permutations, bandwidths fixed).
\begin{table}[h]
    \centering
    \caption{First-stage fit}
    \label{tab:fs}
""" + first_stage_table() + r"""
\end{table}
The instrument is informative in both samples (permutation \(p\leq 0.03\)). In the cross-section the kernel first stage explains about \(6\%\) of the variation in \(A\) out of sample, against \(17\%\) in the panel. The linear first-stage \(F\)-statistics are larger because they test a single coefficient in-sample, not out-of-sample fit.

\subsection*{Results}
\Cref{tab:cross,tab:panel} compare the NPIV average derivative with the linear 2SLS estimate of the same specification. The 2SLS numbers match the paper in the cross-section; in the panel they are the cell-mean analogue of Table A.4, not its respondent-level estimates.
\begin{table}[h]
    \centering
    \caption{Cross-section (\(n = 101\)): linear 2SLS and NPIV}
    \label{tab:cross}
""" + results_table("cross") + r"""
\end{table}
\begin{table}[h]
    \centering
    \caption{Panel (\(n = 1{,}010\)): linear 2SLS and NPIV}
    \label{tab:panel}
""" + results_table("panel") + r"""
\end{table}
Using step sizes of 0.25 and 1 standard deviation instead of 0.5 changes no NPIV estimate by more than """ + f"{max(step_range('cross'), step_range('panel')):.2f}" + r""" (in outcome units per unit of \(A\)).

\subsection*{Reading the results}
\begin{itemize}
    \item The NPIV average derivatives are far smaller than the linear 2SLS estimates, and in all twelve outcome-sample pairs the 2SLS estimate lies outside the NPIV 95\% interval. For example, in the panel the NPIV interval for the concerns index is \([0.10, 4.62]\) and the 2SLS estimate is \(10.47\).
    \item This should not be read as evidence that the effect is near zero. The NPIV fit is shrunk toward a flat function, and in the cross-section cross-validation selects ridge penalties of 10--100 for the three beliefs outcomes. In a simulation with a known slope of 10 and \(n = 1{,}010\), the same estimator returned 9.8 when the instrument explained 81\% of the variance of \(A\) but only 3.5 when it explained 16\%. The panel's kernel first stage has a similar leave-one-out \(R^2\) (0.17) yet the shrinkage in the data is stronger than in that simulation, and we have not diagnosed why.
    \item The bootstrap intervals hold the tuning parameters fixed and so do not capture shrinkage bias. They should be read as sampling variability of a regularized estimator.
    \item There is no accepted pretest like \(F > 10\) for weak instruments in the nonparametric setting, because identification depends on how quickly the singular values of the conditional-expectation operator decay. The linear \(F\)-statistics above do not rule out weak identification of \(\tilde f\).
    \item Not run: the homicide outcomes (Tables 5 and 10 of the paper), which use confidential data not included in the replication package.
\end{itemize}
% END NPIV RESULTS
"""

s = open(TEX).read()
s = s.replace("{../../../../adenc}", "{../../../adenc}")
s = s.replace("= f(a, Z)- f(a', X).", "= f(a, X)- f(a', X).")
if "% BEGIN NPIV RESULTS" in s:
    s = re.sub(r"% BEGIN NPIV RESULTS.*?% END NPIV RESULTS\n", lambda m: SECTION, s, flags=re.S)
else:
    s = s.replace("\\end{document}", SECTION + "\n\\end{document}")
open(TEX, "w").write(s)
print("wrote", TEX)
