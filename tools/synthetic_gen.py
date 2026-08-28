"""
generate_synthetic.py — produce realistic fake data for local pipeline testing.

    python tools/generate_synthetic.py --n-transfers 100000

Produces five files matching the real schema:
    transfers.json          all 4 provider variants with their real field differences
    users.json              referenced by transfers
    recipients.json         referenced by transfers
    transfer_flags.json     APPEND-ONLY history (flag -> unflag -> re-flag)
    partner_chargebacks.csv FOLIOs matching the correct partner-reference per provider

DESIGN NOTES
------------
This is ~90% hand-written logic. Faker is used only for the handful of fields where
human-readable output helps when eyeballing output; those fields are all DROP in the
schema contract anyway, so their content is irrelevant to the model. Faker values are
POOLED (a few thousand generated once, then sampled) because per-row Faker calls
dominate runtime at scale — and pooling is more realistic anyway, since real users
appear across many transactions.

Everything that actually matters comes from numpy: amount distributions, timestamp
spacing, categorical cardinality, and the correlations that make fraud learnable.

LABEL STRUCTURE reproduces the real blind spot:
    ~40% of fraud is caught by rules      -> flag document (latest action = "flag")
    ~60% of fraud is NEVER caught         -> chargeback ONLY (invisible to the rules)
    some legitimate transfers are flagged -> then unflagged (false positives)
This is what lets you test label precedence and see a realistic COVERAGE figure.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

try:
    from faker import Faker
    HAVE_FAKER = True
except ImportError:
    HAVE_FAKER = False


# ── deterministic id helpers ─────────────────────────────────────────────────
def make_oid(namespace: str, i: int) -> str:
    """24-char hex, deterministic — same seed gives same ids across runs."""
    return hashlib.md5(f"{namespace}:{i}".encode()).hexdigest()[:24]


def iso(dt: datetime) -> dict:
    """Mongo extended-JSON date, ISO form (as seen on transfer docs)."""
    return {"$date": dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond//1000:03d}Z"}


def epoch_ms(dt: datetime) -> dict:
    """Mongo extended-JSON date, $numberLong form (as seen on flag docs).
    Deliberately different from iso() so both unwrap paths get exercised."""
    return {"$date": {"$numberLong": str(int(dt.timestamp() * 1000))}}


# ── reference data ───────────────────────────────────────────────────────────
CORRIDORS = [
    ("United States", "Ghana", "USD", "GHS", 11.5),
    ("United States", "Nigeria", "USD", "NGN", 1550.0),
    ("United Kingdom of Great Britain and Northern Ireland", "Ghana", "GBP", "GHS", 14.2),
    ("United Kingdom of Great Britain and Northern Ireland", "Kenya", "GBP", "KES", 165.0),
    ("Canada", "Nigeria", "CAD", "NGN", 1130.0),
    ("Canada", "Ghana", "CAD", "GHS", 8.4),
]
CORRIDOR_P = [0.30, 0.28, 0.16, 0.08, 0.10, 0.08]

PROVIDERS = ["omnex", "truelayer", "checkout", "nmi"]
PROVIDER_P = [0.35, 0.25, 0.22, 0.18]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
                 "icloud.com", "aol.com", "protonmail.com", "mail.com"]
EMAIL_P = [0.46, 0.18, 0.12, 0.09, 0.07, 0.04, 0.025, 0.015]

BANKS = ["POLARIS BANK", "United Bank for Africa", "GTBank", "Zenith Bank",
         "Access Bank", "Ecobank Ghana", "Fidelity Bank Ghana", "Stanbic Bank",
         "Equity Bank Kenya", "KCB Bank", "First Bank of Nigeria", "CalBank"]

NETWORKS = ["mtn", "vodafone", "airteltigo", "safaricom"]
IP_CITIES = ["Accra", "Lagos", "London", "New York", "Toronto", "Kumasi",
             "Abuja", "Nairobi", "Manchester", "Houston"]
IP_COUNTRIES = {"Accra": ("Ghana", "GH"), "Kumasi": ("Ghana", "GH"),
                "Lagos": ("Nigeria", "NG"), "Abuja": ("Nigeria", "NG"),
                "Nairobi": ("Kenya", "KE"), "London": ("United Kingdom", "GB"),
                "Manchester": ("United Kingdom", "GB"),
                "New York": ("United States", "US"), "Houston": ("United States", "US"),
                "Toronto": ("Canada", "CA")}
ASNS = ["AS30986", "AS29465", "AS15169", "AS7922", "AS812", "AS36924"]
RELATIONSHIPS = ["Friend", "Family", "Parent", "Sibling", "Spouse", "Business"]
OCCUPATIONS = ["Dancer", "Engineer", "Nurse", "Teacher", "Driver", "Trader",
               "Accountant", "Student", "Retired", "Consultant"]
DESC_CODES = ["10", "11", "20", "21", "30", "64383d62a8649c1fe6386962"]  # incl. legacy ObjectId
DESC_P = [0.30, 0.20, 0.18, 0.12, 0.10, 0.10]


def build_pools(rng, n_pool=4000):
    """Faker is slow per-call; generate a pool once and sample from it.
    All of these land in DROP fields — realism here is purely for readability."""
    if HAVE_FAKER:
        fake = Faker()
        Faker.seed(int(rng.integers(0, 2**31)))
        return {
            "names": [fake.name().upper() for _ in range(n_pool)],
            "streets": [fake.street_address() for _ in range(n_pool // 4)],
            "cities": [fake.city() for _ in range(n_pool // 8)],
            "phones": [fake.numerify("##########") for _ in range(n_pool)],
        }
    return {
        "names": [f"TEST USER {i:05d}" for i in range(n_pool)],
        "streets": [f"{i} Example Road" for i in range(n_pool // 4)],
        "cities": [f"City{i}" for i in range(n_pool // 8)],
        "phones": [f"{5550000000 + i}" for i in range(n_pool)],
    }


# ── generators ───────────────────────────────────────────────────────────────
def gen_users(rng, pools, n_users, t0):
    users = []
    for i in range(n_users):
        created = t0 - timedelta(days=float(rng.uniform(30, 900)))
        level = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.20, 0.10]))
        limit = {1: 1000, 2: 5000, 3: 15000, 4: 50000}[level]
        used = float(rng.uniform(0, limit))
        name = pools["names"][int(rng.integers(len(pools["names"])))]
        users.append({
            "_id": {"$oid": make_oid("user", i)},
            "customer_id": f"{rng.integers(10**9, 10**10)}",
            # ── sensitive fields: present so the leakage canary has targets ──
            "email": f"user{i}@{rng.choice(EMAIL_DOMAINS, p=EMAIL_P)}",
            "password": "$2b$10$" + hashlib.md5(f"pw{i}".encode()).hexdigest(),
            "pin": "$2b$10$" + hashlib.md5(f"pin{i}".encode()).hexdigest(),
            "ssn": f"{rng.integers(100000000, 999999999)}",
            "full_name": name,
            "first_name": name.split()[0],
            "last_name": name.split()[-1],
            "phone_number": pools["phones"][int(rng.integers(len(pools["phones"])))],
            "state_id_num": f"A{rng.integers(10000000, 99999999)}",
            "plaid_income_user_token": f"user-sandbox-{make_oid('plaid', i)}",
            "billing_address": (f"{pools['streets'][int(rng.integers(len(pools['streets'])))]}, "
                                f"{rng.choice(['US','CA','GB'])}"),
            # ── fields the model actually uses ──
            "isBusinessAccount": bool(rng.random() < 0.06),
            "sender_type": "INDIVIDUAL",
            "signup_country": str(rng.choice(["United States", "Ghana", "United Kingdom",
                                              "Canada", "Nigeria"])),
            "country": str(rng.choice(["US", "GB", "CA"], p=[0.6, 0.25, 0.15])),
            "occupation": str(rng.choice(OCCUPATIONS)),
            "dateCreated": iso(created),
            "dob": iso(datetime(1960, 1, 1, tzinfo=timezone.utc)
                       + timedelta(days=float(rng.uniform(0, 16000)))),
            "threshold_level_no": level,
            "threshold_level": round(used, 2),
            "remainingBalance": round(limit - used, 2),
            "phone_carrier_type": str(rng.choice(["mobile", "voip", "landline"],
                                                 p=[0.86, 0.10, 0.04])),
            "is_phone_verified": "valid",
            "isVerified": bool(level >= 2),
            "proof_of_funds_verification_status": str(
                rng.choice(["approved", "pending", "denied"], p=[0.7, 0.25, 0.05])),
            "addressChangeCounter": int(rng.poisson(0.4)),
            "cardDeletionCounter": int(rng.poisson(0.8)),
            "isDocumentExpired": bool(rng.random() < 0.05),
            "pinFailedAttemptsCount": int(rng.poisson(0.2)),
            "isPinLocked": bool(rng.random() < 0.02),
            # ── leakage fields: must be dropped by the contract ──
            "flag_status": bool(rng.random() < 0.05),
            "date_flagged": iso(created + timedelta(days=30)),
            "isMonitored": bool(rng.random() < 0.03),
            "avs_exempt": False,
            "aml_ctf_status": True,
            "__v": 0,
        })
    return users


def gen_recipients(rng, pools, n_recipients, n_users, t0):
    recips = []
    for i in range(n_recipients):
        created = t0 - timedelta(days=float(rng.exponential(180)))
        country = str(rng.choice(["GH", "NG", "KE"], p=[0.45, 0.40, 0.15]))
        acct = str(rng.choice(["mobile_money", "bank"], p=[0.62, 0.38]))
        name = pools["names"][int(rng.integers(len(pools["names"])))]
        recips.append({
            "_id": {"$oid": make_oid("recip", i)},
            "recipient_id": f"RCPT-{rng.integers(1000000000, 9999999999)}",
            "user_id": make_oid("user", int(rng.integers(n_users))),
            "account_type": acct,
            "bank_name": str(rng.choice(BANKS)) if acct == "bank" else None,
            "bank_code": f"{rng.integers(10000, 99999)}",
            "country": country,
            "status": str(rng.choice(["active", "inactive"], p=[0.94, 0.06])),
            "relationship": str(rng.choice(RELATIONSHIPS,
                                           p=[0.28, 0.30, 0.14, 0.12, 0.10, 0.06])),
            "payoutPartner": str(rng.choice(["omnex", "machnet"], p=[0.7, 0.3])),
            "date_created": iso(created),
            # sensitive — dropped by contract
            "name": name,
            "recipient_first_name": name.split()[0],
            "recipient_last_name": name.split()[-1],
            "recipient_email": f"recip{i}@{rng.choice(EMAIL_DOMAINS, p=EMAIL_P)}",
            "phone_number": pools["phones"][int(rng.integers(len(pools["phones"])))],
            "account_number": f"{rng.integers(10**9, 10**10)}",
            "recipient_city": pools["cities"][int(rng.integers(len(pools["cities"])))],
            "flag_status": False,
            "isDelete": False,
            "__v": 0,
        })
    return recips


def build_transfer(rng, pools, i, user_idx, recip, ts, amount, provider,
                   is_fraud, first_send, corridor_i):
    """Assemble one transfer document with PROVIDER-SPECIFIC fields.

    The field variance here mirrors the real exports: 29 of 86 fields differ by
    paymentProvider, which is exactly what the canonical-field resolution in the
    extractor exists to handle."""
    send_from, send_to, cur_from, cur_to, rate = CORRIDORS[corridor_i]

    # fraud is more likely to come from an odd IP location — this is a signal the
    # model should be able to learn
    if is_fraud and rng.random() < 0.45:
        ip_city = str(rng.choice(["Lagos", "Accra", "Nairobi", "Abuja"]))
    else:
        home = {"United States": ["New York", "Houston"],
                "Canada": ["Toronto"],
                "United Kingdom of Great Britain and Northern Ireland":
                    ["London", "Manchester"]}[send_from]
        ip_city = str(rng.choice(home)) if rng.random() < 0.82 else str(rng.choice(IP_CITIES))
    ip_country, ip_cc = IP_COUNTRIES[ip_city]

    vpn_p = 0.22 if is_fraud else 0.02
    doc = {
        "_id": {"$oid": make_oid("txn", i)},
        "send_amount": round(amount, 2),
        "transfer_fee": round(float(rng.choice([0, 0, 2.5, 3.99])), 2),
        "rate": rate,
        "amount_converted": round(amount, 2),
        "discount": 0,
        "amount_paid": round(amount, 2),
        "sending_currency": cur_from,
        "receive_amount": round(amount * rate, 2),
        "receiving_currency": cur_to,
        "equivalent_send_amount": round(amount * rate, 2),
        "equivalent_transfer_fee": 0,
        "sender_name": pools["names"][int(rng.integers(len(pools["names"])))],
        "sender_phone_number": pools["phones"][int(rng.integers(len(pools["phones"])))],
        "sender_email": f"user{user_idx}@{rng.choice(EMAIL_DOMAINS, p=EMAIL_P)}",
        "sending_from": send_from,
        "sending_to": send_to,
        "delivery_method": recip["account_type"],
        "description": "Family Maintenance",
        "description_code": str(rng.choice(DESC_CODES, p=DESC_P)),
        "recipient_id": recip["_id"]["$oid"],
        "recipient_name": recip["name"],
        "recipient_acct_type": recip["account_type"],
        "recipient_phone_number": recip["phone_number"],
        "payment_method": str(rng.choice(["card", "payment_link"], p=[0.66, 0.34])),
        "status": "pending",
        "payment_status": str(rng.choice(["pending", "success", "hold"], p=[0.2, 0.7, 0.1])),
        "receipt_number": f"JMT-{int(ts.timestamp()*1000)}",
        "user_id": {"$oid": make_oid("user", user_idx)},
        "source": str(rng.choice(["web", "mobile"], p=[0.38, 0.62])),
        "paymentWebhooks": [],
        "payoutWebhooks": [],
        "flag_status": False,
        "ipDetails": {
            "ip": f"154.160.{rng.integers(0,255)}.{rng.integers(0,255)}",
            "security": {
                "vpn": bool(rng.random() < vpn_p),
                "proxy": bool(rng.random() < vpn_p * 0.6),
                "tor": bool(rng.random() < vpn_p * 0.15),
                "relay": bool(rng.random() < 0.01),
            },
            "location": {
                "city": ip_city, "country": ip_country, "country_code": ip_cc,
                "region": "Region", "continent": "Africa",
                "latitude": str(round(float(rng.uniform(-30, 55)), 4)),
                "longitude": str(round(float(rng.uniform(-20, 40)), 4)),
            },
            "network": {"autonomous_system_number": str(rng.choice(ASNS)),
                        "autonomous_system_organization": "SCANCOM"},
        },
        "is_recipient_changed": bool(rng.random() < (0.10 if is_fraud else 0.02)),
        "previous_recipient": [] if rng.random() > 0.08 else [make_oid("recip", int(rng.integers(100)))],
        "retry": 3,
        "paymentProvider": provider,
        "date": iso(ts),
        "__v": 0,
    }
    if recip["account_type"] == "mobile_money":
        doc["recipient_network"] = str(rng.choice(NETWORKS))

    # ── provider-specific fields (this is the schema variance) ──────────────
    if provider == "omnex":
        doc["omnexPaymentId"] = f"JP{rng.integers(10**10, 10**11)}"      # the chargeback FOLIO
        doc["omnexPaymentMtn"] = str(rng.integers(10**9, 10**10))
        doc["transaction_id"] = doc["omnexPaymentMtn"]
        doc["payout_partner_status"] = "other"
        doc["isPendingPayment"] = False
        doc["receiptUrl"] = f"https://orders.example.com/r/{make_oid('rcpt', i)[:8]}"
        doc["omnexCalculateTransferResult"] = {"sendingPrincipal": round(amount, 2)}
    elif provider == "truelayer":
        pr = make_oid("tl", i)
        doc["payment_receipt_number"] = pr                              # the FOLIO here
        doc["transaction_id"] = pr
        doc["paymentLinkUrl"] = f"https://payment.example.com/#id={pr}&resource_token=eyJhbGci"
        doc["socketID"] = make_oid("sock", i)[:12]
    elif provider == "checkout":
        doc["transaction_id"] = f"pay_{make_oid('cko', i)}"
        doc["network_name"] = str(rng.choice(["Visa", "Mastercard"], p=[0.68, 0.32]))
        doc["last_account_num"] = f"{rng.integers(1000, 9999)}"
        doc["risk_score"] = None
        doc["isEnabled3Ds"] = bool(rng.random() < 0.5)
        doc["determine3dsResult"] = {"shouldChallenge": bool(rng.random() < 0.3),
                                     "attemptNo3ds": True}
        doc["account_id"] = f"src_{make_oid('src', i)}"
        doc["scenarioId"] = 1
    else:  # nmi
        doc["transaction_id"] = str(rng.integers(10**10, 10**11))
        doc["network_name"] = str(rng.choice(["Visa", "Mastercard"], p=[0.7, 0.3]))
        doc["last_account_num"] = f"{rng.integers(1000, 9999)}"
        doc["avs_code"] = str(rng.choice(["Y", "N", "X", "A"], p=[0.72, 0.14, 0.09, 0.05]))
        doc["verifiedStatus3Ds"] = str(rng.choice(["passed", "bypassed", "failed"],
                                                  p=[0.62, 0.30, 0.08]))
        doc["isChallenged3Ds"] = bool(rng.random() < 0.25)
        doc["authenticationStatus3Ds"] = None
        doc["determine3dsResult"] = {"shouldChallenge": False, "attemptNo3ds": True}
        if recip["account_type"] == "bank":
            doc["recipient_bank_name"] = recip["bank_name"]
            doc["recipient_bank_code"] = recip["bank_code"]
            doc["recipient_account_number"] = recip["account_number"]
    return doc


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-transfers", type=int, default=100_000)
    ap.add_argument("--n-users", type=int, default=None, help="default: n_transfers // 12")
    ap.add_argument("--n-recipients", type=int, default=None, help="default: n_users * 2")
    ap.add_argument("--fraud-rate", type=float, default=0.02)
    ap.add_argument("--caught-rate", type=float, default=0.40,
                    help="share of fraud the rules engine catches (rest is chargeback-only)")
    ap.add_argument("--false-positive-rate", type=float, default=0.03,
                    help="share of LEGIT transfers flagged then unflagged on review")
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    n = args.n_transfers
    n_users = args.n_users or max(50, n // 12)
    n_recips = args.n_recipients or n_users * 2
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_end = datetime(2026, 6, 1, tzinfo=timezone.utc)
    t_start = t_end - timedelta(days=30 * args.months)

    print(f"Generating {n:,} transfers / {n_users:,} users / {n_recips:,} recipients")
    pools = build_pools(rng)
    if not HAVE_FAKER:
        print("  (faker not installed — using placeholder names; content is irrelevant "
              "since these fields are all DROP)")

    users = gen_users(rng, pools, n_users, t_start)
    recips = gen_recipients(rng, pools, n_recips, n_users, t_start)

    # ── pre-compute per-transfer arrays (vectorised, memory-light) ──────────
    # Zipf-ish user activity: most users transact rarely, a few very often.
    user_weight = rng.pareto(1.6, n_users) + 1
    user_weight /= user_weight.sum()
    user_idx = rng.choice(n_users, size=n, p=user_weight)

    # timestamps sorted so velocity windows and temporal splits are meaningful
    span = (t_end - t_start).total_seconds()
    offsets = np.sort(rng.uniform(0, span, size=n))

    amounts = np.clip(rng.lognormal(4.3, 1.15, size=n), 5, 25000)
    providers = rng.choice(PROVIDERS, size=n, p=PROVIDER_P)
    corridors = rng.choice(len(CORRIDORS), size=n, p=CORRIDOR_P)

    # fraud correlates with amount so the model has something learnable
    amt_rank = amounts.argsort().argsort() / n
    fraud_score = 0.55 * amt_rank + 0.45 * rng.random(n)
    is_fraud = fraud_score > np.quantile(fraud_score, 1 - args.fraud_rate)

    # recipient choice: fraud skews toward brand-new recipients
    recip_idx = rng.choice(n_recips, size=n)
    fresh = rng.choice(n_recips, size=n)
    recip_idx = np.where(is_fraud & (rng.random(n) < 0.6), fresh, recip_idx)

    seen_pairs = set()
    flags, chargebacks = [], []
    caught = is_fraud & (rng.random(n) < args.caught_rate)
    false_pos = (~is_fraud) & (rng.random(n) < args.false_positive_rate)

    big = n > 10_000
    path = out / "transfers.json"
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        if not big:
            fh.write("[\n")
        for i in range(n):
            ts = t_start + timedelta(seconds=float(offsets[i]))
            u, r = int(user_idx[i]), int(recip_idx[i])
            pair = (u, r)
            first = pair not in seen_pairs
            seen_pairs.add(pair)

            doc = build_transfer(rng, pools, i, u, recips[r], ts, float(amounts[i]),
                                 str(providers[i]), bool(is_fraud[i]), first,
                                 int(corridors[i]))

            line = json.dumps(doc)
            fh.write(line + ("\n" if big else (",\n" if i < n - 1 else "\n")))
            written += 1

            # ── labels ──────────────────────────────────────────────────────
            txn_oid = doc["_id"]["$oid"]
            if caught[i]:
                # rules caught it, review upheld -> confirmed fraud
                flags.append({"_id": {"$oid": make_oid("flag", len(flags))},
                              "action": "flag", "transfer": {"$oid": txn_oid},
                              "user": {"$oid": make_oid("user", u)},
                              "admin": None, "isActive": True,
                              "date": epoch_ms(ts + timedelta(hours=2)), "__v": 0})
            elif false_pos[i]:
                # flagged, then cleared on review -> APPEND-ONLY pair, latest wins
                flags.append({"_id": {"$oid": make_oid("flag", len(flags))},
                              "action": "flag", "transfer": {"$oid": txn_oid},
                              "user": {"$oid": make_oid("user", u)},
                              "admin": None, "isActive": True,
                              "date": epoch_ms(ts + timedelta(hours=1)), "__v": 0})
                flags.append({"_id": {"$oid": make_oid("flag", len(flags))},
                              "action": "unflag", "transfer": {"$oid": txn_oid},
                              "user": {"$oid": make_oid("user", u)},
                              "admin": {"$oid": make_oid("admin", 1)}, "isActive": True,
                              "date": epoch_ms(ts + timedelta(hours=26)), "__v": 0})

            if is_fraud[i] and not caught[i]:
                # THE BLIND SPOT: fraud the rules never saw, only a chargeback reveals it
                prov = str(providers[i])
                folio = (doc.get("omnexPaymentId") or doc.get("payment_receipt_number")
                         or doc.get("transaction_id"))
                chargebacks.append({
                    "AGENT_CODE": 111111, "AGENT_DBA": "JUPAY",
                    "DEPOSIT_DATE": (ts + timedelta(days=21)).strftime("%m/%d/%y"),
                    "AMOUNT": -round(float(amounts[i]), 2),
                    "DEPOSIT_NOTE": f"Chargeback {prov}",
                    "UNIQUE TRNX ID @MTN": doc.get("omnexPaymentMtn") or doc["transaction_id"],
                    "FOLIO (TRNX ID AT JUPAY) -common ID between Jupay and partner": folio,
                    "SENDER_NAME": doc["sender_name"].split()[0],
                    "SENDER_MIDDLENAME": "",
                    "SENDER_LASTNAME": doc["sender_name"].split()[-1],
                    "SENDING_PRINCIPAL": round(float(amounts[i]), 2),
                    "SENDING_DATE": ts.strftime("%m/%d/%y"),
                    "RB_DATE": (ts + timedelta(days=21)).strftime("%m/%d/%y"),
                    "Observations // Patterns": str(rng.choice([
                        "New customer. No ID in system. Error code 43 - Stolen Card.",
                        "Unauthorized transaction reported by cardholder.",
                        "Romance scam reported by victim.",
                        "Account takeover - credentials compromised.",
                        "First party misuse - not recognised by customer.",
                    ])),
                    "Watch List / WL": "YES (Customer)",
                })
        if not big:
            fh.write("]\n")

    def dump(obj, name):
        p = out / name
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)
        return p

    dump(users, "users.json")
    dump(recips, "recipients.json")
    dump(flags, "transfer_flags.json")

    cb_path = out / "partner_chargebacks.csv"
    if chargebacks:
        with open(cb_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(chargebacks[0].keys()))
            w.writeheader()
            w.writerows(chargebacks)

    n_fraud = int(is_fraud.sum())
    print(f"\n  transfers.json           {written:,} ({'NDJSON' if big else 'JSON array'})")
    print(f"  users.json               {len(users):,}")
    print(f"  recipients.json          {len(recips):,}")
    print(f"  transfer_flags.json      {len(flags):,} events")
    print(f"  partner_chargebacks.csv  {len(chargebacks):,} rows")
    print(f"\n  true fraud:              {n_fraud:,} ({100*n_fraud/n:.2f}%)")
    print(f"    caught by rules:       {int(caught.sum()):,}")
    print(f"    chargeback only:       {len(chargebacks):,}  <- the blind spot")
    print(f"  false positives:         {int(false_pos.sum()):,} (flag -> unflag)")
    exp_cov = (int(caught.sum()) + int(false_pos.sum()) + len(chargebacks)) / n
    print(f"  expected label coverage: {100*exp_cov:.1f}%")


if __name__ == "__main__":
    main()