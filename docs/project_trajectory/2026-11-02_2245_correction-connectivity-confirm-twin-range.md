## 2026-11-02 22:45 — Correction to 2026-11-02_2232: the range of twin check T' was misquoted

Corrects `2026-11-02_2232_connectivity-confirm-result.md`, section "Checks". That entry says check T'
passes for all 26 CDDM networks "2e-6 to 5e-5, against a limit of 1e-3". The range was read from the
last lines of the run log, not from the output file. In `data/connectivity_confirm.json`, T' over the
26 CDDM networks is 1.7e-6 to 1.3e-4; the largest values are #12 (970815651, 1.3e-4) and #11
(2490685872, 1.1e-4). All 26 remain below the limit of 1e-3, so no network changes status and no
verdict changes. Check T over the four networks with stored row norms is 3.1e-7 to 3.3e-6, as stated.
The Supplementary Methods text of PR r12-weights quotes the corrected bound (1.3e-4).
