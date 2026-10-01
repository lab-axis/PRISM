# PRISM initial feature pool — research basis

## Selection philosophy
The ICI target is defined conservatively. Features are collected broadly first and are
not restricted to official statistics when a reproducible, economically meaningful
market proxy exists. Final retention is deferred to train-only selection, regularization,
and ablation.

## Source rationale
- BLS QCEW: exact industry employment; target ingredient is exposed only with a conservative lag.
- BLS CES: monthly industry employment, hours, earnings, payroll and labor-composition measures.
- BLS PPI: industry net-output prices, i.e. prices received for output sold outside the industry.
- BLS JOLTS: job openings, hires, quits, layoffs/discharges and separations as labor-demand/turnover signals.
- BLS CPI/CPS/CES/PPI national series: common inflation, labor-slack and activity context.
- NY Fed EFFR: observed monetary-policy implementation rate.
- Yahoo Finance: reproducible observed market prices for equities, rates proxies, FX, energy,
  industrial metals, precious metals and broad sector ETFs. These are treated as predictive
  market context, not as authoritative definitions of industry state.

## High-dimensional-predictor rationale
FRED-MD and the Stock-Watson many-predictor literature explicitly motivate broad information
sets followed by factor extraction / variable selection rather than narrow hand-picking.
PRISM follows that philosophy at the candidate stage while preserving out-of-sample selection.

## Timing / leakage
- Fixed quarter lags are not used.
- Every predictor is admitted only when its revision-safe availability date is on or before the quarter-end forecast origin.
- CPI and CPS use unadjusted series; CPI-U/W unadjusted observations are final when issued.
- CES uses unadjusted current-final observations only after the following annual benchmark release.
- PPI uses unadjusted current-final observations only after the four-month revision window closes.
- JOLTS uses unadjusted current-final observations only after the five-year annual revision window closes.
- QCEW current-final observations are exposed only after the following year's Q1 full-data release finalizes that calendar year.
- NY Fed EFFR follows its next-business-day publication rule; market data are available through the quarter-end close.
- Market predictors use raw Close, never retrospectively adjusted Close.
- Scaling and final feature selection are deferred and must be fit on training data only.

## Coverage policy
Structural non-publication for a node is retained as an explicit mask when the same economic
concept is published for a substantial subset of industries. This deliberately replaces the old
"56/56 or exclude" rule.
