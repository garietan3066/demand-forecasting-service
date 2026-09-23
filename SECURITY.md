# Security and experiment integrity

This is a local research prototype, not a production-secured service. No public API is currently implemented. Report security concerns privately through the repository's GitHub security reporting feature if enabled; do not publish credentials or exploit payloads in public issues.

## Current controls

- Raw and processed data stay under ignored `data/`; only code and aggregate experiment reports are committed.
- Baseline execution is offline and reads only training Arrow files named in the audit manifest. It verifies SHA-256 hashes and rejects paths resolving outside the local cache. These hashes detect changes relative to the audit, not publisher authenticity or a maliciously modified manifest.
- No downloaded Python code is executed by the baseline; no model artifact is deserialized by this step.
- CI runs tests and pip-audit for known dependency vulnerabilities with read-only repository permissions and no application credentials. A passing scan is not a guarantee of security.
- Tests check that future/current sales cannot influence present predictions, product histories remain separate, and missing dates or duplicate keys fail validation.

## Experiment boundary

The final published evaluation split is not read by the baseline. The development validation period is June 12–25, 2024. Series selection uses IDs present on March 28 only. Every prediction uses earlier observations; actuals from earlier validation dates may be used once those dates have passed. Stockout status on the target date is used only to segment evaluation, never as a forecast input. Future observed weather is excluded.

Stockout-hour sales are preserved, not automatically imputed or erased. Publisher descriptions do not fully establish within-hour event timing or the interpretation of discount values above one. Those fields are not prediction inputs in this baseline. Validation against observed sales does not establish recovery of latent demand.

## Before deployment

Add authentication, HTTPS, request limits, secret scanning, protected branches, reviewed dependency locks, and container hardening. Secrets belong in environment variables or a secret manager. Load joblib artifacts only from a trusted build process; never accept client-provided model files. Pin release dependencies and review CI action versions before a production release.
