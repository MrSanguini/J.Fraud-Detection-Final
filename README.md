# Fraud Data Pipeline

Turns raw MongoDB exports (transfers, users, recipients, flags) plus partner
chargeback sheets into a single pseudonymised training table.

**Runs entirely locally. No data leaves the machine.**

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

# one-time: create .env in the project root (NEVER commit it)
#   FRAUD_SALT=<paste a long random string>
# generate one with:
python -c "import secrets; print(secrets.token_hex(32))"

python src/run_pipeline.py --chargebacks data/partner_chargebacks.csv
```

Put raw exports in `data/`:

| File | Contents |
|---|---|
| `transfers.json` (or per-provider: `checkout_transfers.json`, `nmi_transfers.json`, `omnex_transfers.json`, `transfer.json`) | transfer documents — multiple files are concatenated |
| `users.json` | user documents |
| `recipients.json` | recipient documents |
| `transfer_flags.json` | **transfer**-flag collection only (not user/recipient flags) |
| `partner_chargebacks.csv` | partner chargeback sheet |

Output: `data/training_set.parquet` (or `.csv` if pyarrow is unavailable).

---

## Stages

| | Script | Does |
|---|---|---|
| A | `profile_schema.py` | Detects provider-driven schema variance. **Run on every fresh export.** |
| B | `extractor.py` | Raw JSON → flat tables. Allowlist, hashing, canonical fields. |
| C0 | `chargeback_parser.py` | Partner sheet → confirmed-fraud labels. |
| C | `label_builder.py` | Flag history + chargebacks → per-transaction labels. |
| D | `assembler.py` | Joins + features → training table. |
| E | `test_pipeline.py` | Safety assertions. **Not optional.** |

`schema_contract.py` is the single source of truth every stage reads.

---

## Key design decisions

### Security: exclusion, not obfuscation
Sensitive fields (`ssn`, `password`, `pin`, tokens, names, phone/account numbers,
raw IP, lat/long) are **dropped entirely** — never hashed into the output.
Hashing is used *only* for non-sensitive join keys that must stay linkable.

The contract is an **allowlist**: any field not listed is dropped. A new
sensitive field added upstream tomorrow cannot leak in, because no code path
reads it.

> The output is *pseudonymised*, not anonymous. Keep `FRAUD_SALT` secret and
> treat the training set as a confidential internal asset.

### Provider schema variance
`paymentProvider` is a **discriminator**: its value determines which fields
exist. Confirmed across four variants — **29 of 86 fields vary**. The
partner-side transaction reference is named differently per provider, so
`CANONICAL` coalesces aliases into one column:

```
partner_reference  <- omnexPaymentId | payment_receipt_number | transaction_id
partner_txn_id     <- omnexPaymentMtn | transaction_id
```

Both are extracted to maximise chargeback join coverage (100% on the mocks).

### Labels
Flag documents are **append-only** — a transfer may have a history
(flag → unflag → re-flag). The **latest event by date** is the verdict.

Precedence, highest authority first:
1. **chargeback** (external confirmation) → fraud
2. **flag verdict** → fraud / legit
3. **no flag** → *assumed* legit

A chargeback **overrides** an unflag: review cleared it, the partner charged it
back — a false negative of the review process, and the most valuable training
example available.

### Point-in-time correctness
Every feature for a transaction at time T uses only information available at T:
- rolling windows use `closed="left"` — a transaction never counts in its own velocity aggregate
- recipient history uses `cumcount()` — prior sends only
- user/recipient docs are **not** snapshotted historically, so only
  IMMUTABLE/STATIC fields (creation dates) are trusted on old records

### Two tiers
| Tier | Available on | Examples |
|---|---|---|
| 1 | every record | amount, corridor, VPN/proxy/Tor, IP mismatch, account age, recipient age, 3DS, AVS |
| 2 | future snapshotted records only | account level, limit utilisation, verification state, billing match |

Tier-2 columns are NaN on historical rows and populate automatically once IT's
transfer-time snapshots arrive — no code change needed.

---

## Known limitations

**Label coverage is the bottleneck.** The `COVERAGE` figure in the label audit
is the fraction of transfers with a *confirmed* verdict. Everything else is
assumed-legit and contains the fraud the rules never caught. Train on that alone
and the model learns to **imitate the existing rules**, blind spots included.
Partner chargeback data is what fixes this — it is the highest-priority input.

**Historical records lose Tier-2 signal.** Deliberate: correctness over strength.
Recoverable only if MongoDB oplog/backups reach back far enough (worth one
question to IT; usually they don't).

**Watchlist feedback is missing.** Partners watchlist a user but only notify via
a *later* declined transaction — so the transaction that caused it is unlabeled.
Ask partners to report which transaction triggered a watchlisting.

**`description_code` format is inconsistent** — `"10"` in newer records vs an
ObjectId string in older ones. Needs normalising before it is a usable
categorical.

---

## Adding a new field

1. Add it to the right contract in `schema_contract.py` with action /
   sensitivity / mutability / tier / reason.
2. If it needs transforming, add a derivation in `assembler.py`.
3. Re-run and confirm `test_pipeline.py` still passes.

Fields are dropped by default. That is intentional.
