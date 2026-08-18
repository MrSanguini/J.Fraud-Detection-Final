"""
schema_contract.py — the single source of truth for the fraud-data pipeline.

DECLARATIVE ONLY. No data is read or transformed here. The extractor and feature
assembler import these tables and act on them.

ALLOWLIST BY DESIGN: any field not listed here is dropped. A new sensitive field
added upstream cannot leak into the training data, because no code path reads it.

Four axes per field:
  action      KEEP    -> becomes a feature
              DERIVE  -> not used raw; feeds a computed feature
              HASH    -> pseudonymised join key; never a feature
              LABEL   -> builds the target, then discarded
              META    -> pipeline logic only (timestamps, cost figures)
              DROP    -> excluded; reason recorded for audit

  sensitivity SENSITIVE (PII/secret — never reaches output)
              INTERNAL  (safe to use)

  mutability  IMMUTABLE -> written once at creation; safe for ALL records
              STATIC    -> set-once in practice (creation dates); safe for all
              MUTABLE   -> changes over time; only safe on FUTURE snapshotted records
              NA        -> not applicable

  tier        1 -> available for every historical record
              2 -> requires the transfer-time snapshot IT is adding; included
                   conditionally when present, skipped when absent
"""

# ─────────────────────────────────────────────────────────────────────────────
#  CANONICAL FIELDS — solve the provider schema-variance problem.
#
#  `paymentProvider` is a DISCRIMINATOR: its value determines which other fields
#  exist. Confirmed across 4 provider variants (truelayer/checkout/nmi/omnex):
#  29 of 86 fields vary. The partner-side transaction reference is named
#  differently per provider, so we coalesce aliases into one canonical column.
#
#  Chargeback join mapping (from the partner chargeback sheet):
#     FOLIO ("common ID between us and partner") -> omnexPaymentId  (JP-prefixed)
#     "UNIQUE TRNX ID @<partner>"                -> omnexPaymentMtn / transaction_id
#  We therefore extract BOTH keys to maximise chargeback join coverage.
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL = {
    # primary partner reference — first non-null wins, in this order
    "partner_reference": {
        "sources": ["omnexPaymentId", "payment_receipt_number", "transaction_id"],
        "action": "HASH",
        "reason": "partner-side ID; name varies by paymentProvider. Chargeback FOLIO joins here.",
    },
    # secondary/universal reference — present on all four providers
    "partner_txn_id": {
        "sources": ["omnexPaymentMtn", "transaction_id"],
        "action": "HASH",
        "reason": "universal partner txn id; fallback chargeback join key.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFER DOCUMENT — immutable core. Most Tier-1 signal lives here.
# ─────────────────────────────────────────────────────────────────────────────
TRANSFER = {
    "_id":                  ("HASH",   "INTERNAL",  "IMMUTABLE", 1, "transaction id / label join key"),
    "user_id":              ("HASH",   "INTERNAL",  "IMMUTABLE", 1, "sender join key -> user._id"),
    "recipient_id":         ("HASH",   "INTERNAL",  "IMMUTABLE", 1, "recipient join key -> recipient._id"),

    # ── amounts: two jobs, two fields ───────────────────────────────────────
    "send_amount":          ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "FEATURE: attempted amount, known at score time"),
    "amount_paid":          ("META",   "INTERNAL",  "IMMUTABLE", 1, "COST MODEL ONLY: realised loss; NOT a feature"),
    "transfer_fee":         ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "fee charged"),
    "rate":                 ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "FX rate"),
    "discount":             ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "promo discount"),
    "sending_currency":     ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "categorical"),
    "receiving_currency":   ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "categorical"),
    "equivalent_send_amount": ("DROP", "INTERNAL",  "IMMUTABLE", 0, "= send_amount x rate; linearly dependent"),
    "receive_amount":       ("DROP",   "INTERNAL",  "IMMUTABLE", 0, "post-conversion, redundant"),
    "amount_converted":     ("DROP",   "INTERNAL",  "IMMUTABLE", 0, "redundant with send_amount"),
    "equivalent_transfer_fee": ("DROP","INTERNAL",  "IMMUTABLE", 0, "= transfer_fee x rate"),

    # ── corridor & channel ──────────────────────────────────────────────────
    "sending_from":         ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "origin country (US/CA/UK)"),
    "sending_to":           ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "destination (always African)"),
    "delivery_method":      ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "payout channel"),
    "recipient_acct_type":  ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "bank / mobile_money"),
    "recipient_network":    ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "e.g. mtn (absent on some providers)"),
    "recipient_bank_name":  ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "payout bank (nmi variant)"),
    "payment_method":       ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "card / payment_link"),
    "paymentProvider":      ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "DISCRIMINATOR + genuine risk signal"),
    "source":               ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "web / mobile"),
    "description_code":     ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "purpose-of-transfer categorical (see note: format varies)"),
    "description":          ("DROP",   "SENSITIVE", "IMMUTABLE", 0, "free text; use description_code"),

    # ── card / payment intelligence (provider-variant fields) ───────────────
    "network_name":         ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "card network: Visa/Mastercard (checkout, nmi)"),
    "avs_code":             ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "address verification result — real fraud signal (nmi)"),
    "risk_score":           ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "partner's own auth-time risk score, if supplied"),

    # ── 3-D Secure family ───────────────────────────────────────────────────
    "verifiedStatus3Ds":    ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "pass/fail/bypassed"),
    "isEnabled3Ds":         ("DERIVE", "INTERNAL",  "IMMUTABLE", 1, "-> threeds_reported flag (null = partner silent)"),
    "isChallenged3Ds":      ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "was the user challenged (nmi)"),
    "authenticationStatus3Ds": ("KEEP","INTERNAL",  "IMMUTABLE", 1, "3DS auth status (nmi)"),
    "determine3dsResult.shouldChallenge": ("KEEP","INTERNAL","IMMUTABLE",1,"3DS decision (checkout, nmi)"),
    "determine3dsResult.attemptNo3ds":    ("KEEP","INTERNAL","IMMUTABLE",1,"3DS bypass attempt"),

    # ── recipient-change signals ────────────────────────────────────────────
    "is_recipient_changed": ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "last-minute swap signal"),
    "previous_recipient":   ("DERIVE", "INTERNAL",  "IMMUTABLE", 1, "-> count of prior recipients"),

    # ── IP / device intelligence ────────────────────────────────────────────
    "ipDetails.ip":                    ("DERIVE", "SENSITIVE", "IMMUTABLE", 1, "raw IP: dropped after geo already resolved"),
    "ipDetails.security.vpn":          ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "strong signal"),
    "ipDetails.security.proxy":        ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "strong signal"),
    "ipDetails.security.tor":          ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "strong signal"),
    "ipDetails.security.relay":        ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "strong signal"),
    "ipDetails.location.country_code": ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "IP country categorical"),
    "ipDetails.location.city":         ("KEEP",   "INTERNAL",  "IMMUTABLE", 1, "IP city categorical"),
    "ipDetails.location.country":      ("DERIVE", "INTERNAL",  "IMMUTABLE", 1, "-> IP-vs-corridor mismatch flag"),
    "ipDetails.location.latitude":     ("DROP",   "SENSITIVE", "IMMUTABLE", 0, "precise geo unnecessary over city"),
    "ipDetails.location.longitude":    ("DROP",   "SENSITIVE", "IMMUTABLE", 0, "precise geo"),
    "ipDetails.network.autonomous_system_number": ("KEEP","INTERNAL","IMMUTABLE",1,"ASN; hosting ASNs correlate w/ fraud"),

    # ── email (domain only) ─────────────────────────────────────────────────
    "sender_email":         ("DERIVE", "SENSITIVE", "IMMUTABLE", 1, "-> domain only; address discarded, never written"),

    # ── timestamp ───────────────────────────────────────────────────────────
    "date":                 ("META",   "INTERNAL",  "IMMUTABLE", 1, "point-in-time anchor (confirmed immutable)"),

    # ── LEAKAGE drops: outcomes known only AFTER the scoring moment ─────────
    # Scoring happens BEFORE payment_status is marked (confirmed business rule).
    "status":                 ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: recipient-received, post-score"),
    "payment_status":         ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: sender-paid, post-score"),
    "payment_partner_status": ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: partner outcome, post-score"),
    "payout_partner_status":  ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: payout outcome, post-score"),
    "isPendingPayment":       ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: lifecycle state"),
    "flag_status":            ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: shadow of the label"),
    "date_flagged":           ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: appears on nmi transfers — label shadow"),
    "retry":                  ("DROP", "INTERNAL", "MUTABLE", 0, "post-hoc lifecycle"),
    "retry_receipt_number":   ("DROP", "INTERNAL", "MUTABLE", 0, "post-hoc retries"),
    "paymentWebhooks":        ("DROP", "INTERNAL", "MUTABLE", 0, "post-score settlement events"),
    "payoutWebhooks":         ("DROP", "INTERNAL", "MUTABLE", 0, "post-score settlement events"),
    "paymentWebhookProcessed":("DROP", "INTERNAL", "MUTABLE", 0, "post-score"),
    "payoutWebhookProcessed": ("DROP", "INTERNAL", "MUTABLE", 0, "post-score"),
    "omnexPayoutReceiveEvent":("DROP", "INTERNAL", "MUTABLE", 0, "post-score"),
    "isHoldMailSent":         ("DROP", "INTERNAL", "MUTABLE", 0, "post-score action"),

    # ── SENSITIVE drops ─────────────────────────────────────────────────────
    "sender_name":            ("DROP", "SENSITIVE", "NA", 0, "PII; use hashed user_id"),
    "sender_phone_number":    ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "recipient_name":         ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "recipient_phone_number": ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "recipient_phone_number_intl": ("DROP","SENSITIVE","NA", 0, "PII"),
    "recipient_account_number":("DROP","SENSITIVE", "NA", 0, "bank account number"),
    "last_account_num":       ("DROP", "SENSITIVE", "NA", 0, "card last-4"),
    "paymentLinkUrl":         ("DROP", "SENSITIVE", "NA", 0, "contains JWT"),
    "receiptUrl":             ("DROP", "SENSITIVE", "NA", 0, "customer receipt link"),
    "payout_callback_url":    ("DROP", "SENSITIVE", "NA", 0, "internal token URL"),

    # ── plumbing drops ──────────────────────────────────────────────────────
    "relationship":           ("DROP", "INTERNAL", "NA", 0, "use recipient.relationship (authoritative)"),
    "sourceOfFunds":          ("DROP", "INTERNAL", "NA", 0, "unused per business"),
    "referral_redeemed":      ("DROP", "INTERNAL", "MUTABLE", 0, "not known at score time"),
    "referral_code":          ("DROP", "INTERNAL", "NA", 0, "plumbing"),
    "receipt_number":         ("DROP", "INTERNAL", "NA", 0, "internal ref; partner_reference used instead"),
    "recipient_bank_code":    ("DROP", "INTERNAL", "NA", 0, "redundant with bank_name"),
    "scenarioId":             ("DROP", "INTERNAL", "NA", 0, "test-harness artifact"),
    "licensorHolder":         ("DROP", "INTERNAL", "NA", 0, "integration id"),
    "isDowngraded3Ds":        ("DROP", "INTERNAL", "NA", 0, "always null in samples"),
    "nmi3dsData":             ("DROP", "INTERNAL", "NA", 0, "unused per business"),
    "ach_type":               ("DROP", "INTERNAL", "NA", 0, "unused per business"),
    "interchange_type":       ("DROP", "INTERNAL", "NA", 0, "unused per business"),
    "omnexCalculateTransferResult": ("DROP","INTERNAL","NA",0,"duplicate fee/amount data"),
    "ignorePartnerProvider":  ("DROP", "INTERNAL", "NA", 0, "internal flag"),
    "socketID":               ("DROP", "INTERNAL", "NA", 0, "session plumbing"),
    "account_id":             ("DROP", "INTERNAL", "NA", 0, "provider account ref; often empty"),
    "sender_id":              ("DROP", "INTERNAL", "NA", 0, "duplicate of user_id or provider ref"),
    "pst_created_date":       ("DROP", "INTERNAL", "NA", 0, "duplicate of date, unparsed format"),
    # transaction_id / payment_receipt_number / omnexPayment* consumed by CANONICAL
}

# ─────────────────────────────────────────────────────────────────────────────
#  USER DOCUMENT — Tier 2 unless STATIC (create-once). Documents are NOT
#  snapshotted at transfer time for historical records, so MUTABLE fields are
#  only trustworthy on future records that carry the snapshot.
# ─────────────────────────────────────────────────────────────────────────────
USER = {
    "_id":                  ("HASH",   "INTERNAL",  "IMMUTABLE", 1, "join key"),
    "dateCreated":          ("DERIVE", "INTERNAL",  "STATIC",    1, "-> account age at txn (immutable: safe for all)"),
    "dob":                  ("DERIVE", "SENSITIVE", "STATIC",    1, "-> age at txn (dob immutable: safe)"),
    "isBusinessAccount":    ("KEEP",   "INTERNAL",  "STATIC",    1, "set at creation"),
    "signup_country":       ("KEEP",   "INTERNAL",  "STATIC",    1, "IP-derived at signup, immutable"),
    "occupation":           ("KEEP",   "INTERNAL",  "STATIC",    1, "set at onboarding"),
    "sender_type":          ("KEEP",   "INTERNAL",  "STATIC",    1, "INDIVIDUAL / business"),

    # ── Tier 2: mutable, future snapshotted records only ────────────────────
    "threshold_level_no":   ("KEEP",   "INTERNAL",  "MUTABLE", 2, "account level 1-4; strong signal"),
    "threshold_level":      ("DERIVE", "INTERNAL",  "MUTABLE", 2, "with remainingBalance -> limit utilisation"),
    "remainingBalance":     ("DERIVE", "INTERNAL",  "MUTABLE", 2, "-> limit utilisation"),
    "country":              ("KEEP",   "INTERNAL",  "MUTABLE", 2, "currently-selected country"),
    "phone_carrier_type":   ("KEEP",   "INTERNAL",  "MUTABLE", 2, "voip is a fraud signal"),
    "is_phone_verified":    ("KEEP",   "INTERNAL",  "MUTABLE", 2, "verification state"),
    "isVerified":           ("KEEP",   "INTERNAL",  "MUTABLE", 2, "lvl-2 ID verification passed"),
    "proof_of_funds_verification_status": ("KEEP","INTERNAL","MUTABLE",2,"approved/pending/denied"),
    "addressChangeCounter": ("KEEP",   "INTERNAL",  "MUTABLE", 2, "address churn"),
    "cardDeletionCounter":  ("KEEP",   "INTERNAL",  "MUTABLE", 2, "card churn; fraud signal"),
    "isDocumentExpired":    ("KEEP",   "INTERNAL",  "MUTABLE", 2, "expired ID"),
    "pinFailedAttemptsCount":("KEEP",  "INTERNAL",  "MUTABLE", 2, "failed PIN attempts"),
    "isPinLocked":          ("KEEP",   "INTERNAL",  "MUTABLE", 2, "account locked"),
    "billing_address":      ("DERIVE", "SENSITIVE", "MUTABLE", 2, "-> billing/account country match flag"),

    # ── SENSITIVE drops ─────────────────────────────────────────────────────
    "ssn":                  ("DROP", "SENSITIVE", "NA", 0, "NEVER"),
    "password":             ("DROP", "SENSITIVE", "NA", 0, "secret hash"),
    "pin":                  ("DROP", "SENSITIVE", "NA", 0, "secret hash"),
    "passwordResetToken":   ("DROP", "SENSITIVE", "NA", 0, "secret"),
    "plaid_income_user_token":  ("DROP","SENSITIVE","NA",0,"token"),
    "plaid_asset_report_token": ("DROP","SENSITIVE","NA",0,"token"),
    "notification_token":   ("DROP", "SENSITIVE", "NA", 0, "push token"),
    "email":                ("DROP", "SENSITIVE", "NA", 0, "PII (transfer-level domain used instead)"),
    "emails":               ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "full_name":            ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "first_name":           ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "last_name":            ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "names":                ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "phone_number":         ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "phone_numbers":        ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "addresses":            ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "address":              ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "state_id_num":         ("DROP", "SENSITIVE", "NA", 0, "government ID number"),
    "ip_address":           ("DROP", "SENSITIVE", "NA", 0, "raw IP"),

    # ── LEAKAGE drops ───────────────────────────────────────────────────────
    "flag_status":          ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: user-flag shadow"),
    "date_flagged":         ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE"),
    "isMonitored":          ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE"),
    "date_monitored":       ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE"),
    "flag_exempt":          ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: staff-set exemption"),
    "flag_exemptions":      ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: staff-set"),
    "avs_exempt":           ("DROP", "INTERNAL", "MUTABLE", 0, "LEAKAGE: staff-set"),
    "card_deletion_limit_exempt": ("DROP","INTERNAL","MUTABLE",0,"LEAKAGE: staff-set"),

    # ── deprecated / redundant per business ─────────────────────────────────
    "aml_ctf_status":       ("DROP", "INTERNAL", "MUTABLE", 0, "deprecated"),
    "accountVerified":      ("DROP", "INTERNAL", "MUTABLE", 0, "redundant (= other two)"),
    "emailVerified":        ("DROP", "INTERNAL", "MUTABLE", 0, "auto-true, deprecated"),
    "proof_of_funds_verification": ("DROP","INTERNAL","MUTABLE",0,"redundant per business"),
    "threshold_monthly_levels":    ("DROP","INTERNAL","MUTABLE",0,"redundant"),
    "threshold_six_months_levels": ("DROP","INTERNAL","MUTABLE",0,"redundant"),
    "threshold_fifteen_days_level":("DROP","INTERNAL","MUTABLE",0,"redundant"),
    "threshold_two_months_levels": ("DROP","INTERNAL","MUTABLE",0,"redundant"),
    "previousThreshold":    ("DROP", "INTERNAL", "NA", 0, "ignore per business"),
    "customer_id":          ("DROP", "INTERNAL", "NA", 0, "customer-facing, no meaning"),
    "isPayoutRequestPending":("DROP","INTERNAL","NA", 0, "redundant"),
    "stateId":              ("DROP", "INTERNAL", "NA", 0, "redundant"),
    "licensor":             ("DROP", "INTERNAL", "NA", 0, "integration id"),
    "acceptTerms":          ("DROP", "INTERNAL", "NA", 0, "does not apply"),
    "isBusinessFormCompleted": ("DROP","INTERNAL","NA",0,"redundant"),
    "activateMigrationPrompt": ("DROP","INTERNAL","NA",0,"useless"),
    "forcePlaidRegistration":  ("DROP","INTERNAL","NA",0,"useless"),
    "isProofOfFundsReset":     ("DROP","INTERNAL","NA",0,"ignored"),
    "currentDeviceBuildNumber":("DROP","INTERNAL","NA",0,"always null"),
    "currentDeviceInfo":       ("DROP","INTERNAL","NA",0,"always null"),
    "isLimitNotificationPromptActivated": ("DROP","INTERNAL","NA",0,"useless"),
}

# ─────────────────────────────────────────────────────────────────────────────
#  RECIPIENT DOCUMENT — date_created is STATIC (Tier 1, high signal).
# ─────────────────────────────────────────────────────────────────────────────
RECIPIENT = {
    "_id":                  ("HASH",   "INTERNAL",  "IMMUTABLE", 1, "join key -> transfer.recipient_id"),
    "user_id":              ("HASH",   "INTERNAL",  "IMMUTABLE", 1, "owning user (integrity check)"),
    "date_created":         ("DERIVE", "INTERNAL",  "STATIC",    1, "-> recipient age at txn (HIGH signal)"),
    "country":              ("KEEP",   "INTERNAL",  "STATIC",    1, "recipient country"),
    "account_type":         ("KEEP",   "INTERNAL",  "STATIC",    1, "bank / mobile_money"),
    "bank_name":            ("KEEP",   "INTERNAL",  "STATIC",    1, "frequency-encode; rarity signal"),
    "payoutPartner":        ("KEEP",   "INTERNAL",  "STATIC",    1, "e.g. omnex"),
    "relationship":         ("KEEP",   "INTERNAL",  "MUTABLE", 2, "sender-stated relationship (controlled vocab)"),
    "status":               ("KEEP",   "INTERNAL",  "MUTABLE", 2, "active / inactive"),
    "recipient_email":      ("DERIVE", "SENSITIVE", "STATIC",    1, "-> domain only; address discarded"),

    # ── drops ───────────────────────────────────────────────────────────────
    "recipient_id":         ("DROP", "INTERNAL",  "NA", 0, "RCPT- string; customer-facing, redundant"),
    "account_number":       ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "phone_number":         ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "phone_number_intl":    ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "name":                 ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "recipient_first_name": ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "recipient_last_name":  ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "recipient_city":       ("DROP", "SENSITIVE", "NA", 0, "PII"),
    "bank_code":            ("DROP", "INTERNAL",  "NA", 0, "redundant with bank_name"),
    "flag_status":          ("DROP", "INTERNAL",  "MUTABLE", 0, "LEAKAGE: recipient-flag shadow"),
    "isDelete":             ("DROP", "INTERNAL",  "MUTABLE", 0, "post-hoc state"),
    "relationship_id":      ("DROP", "INTERNAL",  "NA", 0, "redundant with relationship"),
    "omnexRelationshipId":  ("DROP", "INTERNAL",  "NA", 0, "partner id"),
    "externalIds":          ("DROP", "INTERNAL",  "NA", 0, "partner ids"),
    "externalBankDetails":  ("DROP", "SENSITIVE", "NA", 0, "contains account number"),
    "externalPhoneDetails": ("DROP", "SENSITIVE", "NA", 0, "PII"),
}

# ─────────────────────────────────────────────────────────────────────────────
#  FLAG DOCUMENTS (transfer-flag collection) — label + timestamp only.
#  APPEND-ONLY: every flag/unflag creates a NEW document. Latest event wins.
# ─────────────────────────────────────────────────────────────────────────────
FLAG = {
    "transfer":  ("HASH",  "INTERNAL", "IMMUTABLE", 1, "join key -> transfer._id (MUST be hashed with the same salt or the join silently fails)"),
    "action":    ("LABEL", "INTERNAL", "IMMUTABLE", 1, "flag/unflag; latest-by-date is the verdict"),
    "date":      ("META",  "INTERNAL", "IMMUTABLE", 1, "orders the append-only history"),
    "admin":     ("DROP",  "INTERNAL", "NA", 0, "LEAKAGE: system-vs-staff, post-hoc"),
    "reason":    ("DROP",  "SENSITIVE","NA", 0, "free text"),
    "user":      ("DROP",  "INTERNAL", "NA", 0, "not needed for transfer labels"),
    "recipient": ("DROP",  "INTERNAL", "NA", 0, "not a transfer flag"),
    "isActive":  ("DROP",  "INTERNAL", "NA", 0, "does nothing per business"),
}

CONTRACTS = {"transfer": TRANSFER, "user": USER,
             "recipient": RECIPIENT, "flag": FLAG}

# Discriminator fields to group by when profiling schema variance.
DISCRIMINATORS = ["paymentProvider", "delivery_method", "recipient_acct_type", "source"]
