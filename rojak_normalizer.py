import re
import pandas as pd


class RojakRentalNormalizer:
    """
    3-Layer Rojak Normalization + Slot Extraction + Listing Matcher
    for KL & Selangor Residential Property Rental Agent.
    """

    AMBIGUOUS_CROSS_STATE_AREAS = {"cheras", "ampang", "puchong", "gombak", "serdang"}
    KNOWN_AREAS = [
        "Setapak", "Wangsa Maju", "Cheras", "Petaling Jaya", "Cyberjaya",
        "Shah Alam", "Sentul", "Puchong", "Ampang", "Subang Jaya",
        "Kota Damansara", "Mont Kiara", "Seri Kembangan", "Old Klang Road",
        "Bukit Jalil", "Kajang", "Kepong", "Klang"
    ]

    def __init__(self, slang_csv_path="rojak_slang_dict.csv"):
        self.slang_csv_path = slang_csv_path
        self.rules = []
        self.reload_dictionary()

    def reload_dictionary(self):
        df = pd.read_csv(self.slang_csv_path)
        df["term_len"] = df["slang_term"].astype(str).str.len()
        df = df.sort_values(by=["priority", "term_len"], ascending=[True, False])
        self.rules = []
        for _, row in df.iterrows():
            term = str(row["slang_term"]).strip()
            norm = str(row["normalized_text"]).strip()
            mtype = str(row["match_type"]).strip()
            escaped = re.escape(term)
            if mtype == "substring":
                pattern = re.compile(escaped, flags=re.IGNORECASE)
            else:
                # Both 'word_boundary' and 'phrase' enforce non-word boundaries to prevent partial-word corruption
                pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
            self.rules.append({
                "slang_term": term,
                "normalized_text": norm,
                "category": row["category"],
                "match_type": mtype,
                "priority": int(row["priority"]),
                "target_slot": row["target_slot"],
                "pattern": pattern,
            })

    def normalize(self, text: str) -> dict:
        """
        Dual-track normalization: preserves raw_text for tone/language matching
        and outputs normalized_text + matched_slangs for NLU/RAG retrieval.
        """
        raw_text = str(text)
        norm_text = raw_text
        matched = []

        for rule in self.rules:
            if rule["pattern"].search(norm_text):
                norm_text = rule["pattern"].sub(rule["normalized_text"], norm_text)
                matched.append((rule["slang_term"], rule["normalized_text"], rule["category"]))

        norm_text = re.sub(r"\s+", " ", norm_text).strip()
        return {
            "raw_text": raw_text,
            "normalized_text": norm_text,
            "matched_slangs": matched,
        }

    def extract_slots(self, normalized_text: str) -> dict:
        t = normalized_text
        t_low = t.lower()
        slots = {
            "unit_id": None,
            "pv_code": None,
            "area": None,
            "is_cross_state_area": False,
            "max_budget_rm": None,
            "rooms": None,
            "furnished": None,
            "min_parking": None,
            "near_lrt": False,
            "cooking_allowed": False,
        }

        # Unit / Legacy Ref ID (U001-U050 or R001-R005)
        m_id = re.search(r"\b([UR]\d{3})\b", t, flags=re.IGNORECASE)
        if m_id:
            slots["unit_id"] = m_id.group(1).upper()

        # PV Series code
        m_pv = re.search(r"\b(PV\s?\d+)\b", t, flags=re.IGNORECASE)
        if m_pv:
            slots["pv_code"] = re.sub(r"\s+", "", m_pv.group(1).upper())

        # Area detection
        for area in self.KNOWN_AREAS:
            if re.search(rf"(?<!\w){re.escape(area)}(?!\w)", t, flags=re.IGNORECASE):
                slots["area"] = area
                if area.lower() in self.AMBIGUOUS_CROSS_STATE_AREAS:
                    slots["is_cross_state_area"] = True
                break

        # Budget extraction (e.g. RM1800, RM 1,800, bawah 2000, below 1500)
        m_rm = re.search(r"(?:rm\s*|bawah\s*|below\s*|under\s*|budget\s*)(\d{3,4}(?:,\d{3})?)", t_low)
        if m_rm:
            val = int(m_rm.group(1).replace(",", ""))
            if 400 <= val <= 10000:
                slots["max_budget_rm"] = val

        # Rooms extraction (e.g. 3 bilik, 3 rooms, 2 bedroom, 3房)
        m_rooms = re.search(r"\b([1-5])\s*(?:bilik|rooms?|bedrooms?|br)\b|([1-5])\s*房", t_low)
        if m_rooms:
            slots["rooms"] = int(m_rooms.group(1) or m_rooms.group(2))

        # Furnished status
        if "fully furnished" in t_low:
            slots["furnished"] = "Fully Furnished"
        elif "partially furnished" in t_low:
            slots["furnished"] = "Partially Furnished"
        elif "not furnished" in t_low:
            slots["furnished"] = "Not Furnished"

        # Parking count
        m_park = re.search(r"\b([1-3])\s*(?:parking|lots?)\b", t_low)
        if m_park:
            slots["min_parking"] = int(m_park.group(1))

        # Key facilities
        if any(k in t_low for k in ["near ktm/lrt", "lrt", "mrt", "ktm"]):
            slots["near_lrt"] = True
        if "cooking allowed" in t_low or "masak" in t_low or "cooking" in t_low:
            slots["cooking_allowed"] = True

        return slots

    def match_listings(self, listings_df: pd.DataFrame, slots: dict, top_k: int = 3) -> dict:
        df = listings_df.copy()
        relaxed_filters = []

        # Direct ID match takes highest priority
        if slots["unit_id"]:
            uid = slots["unit_id"]
            hit = df[(df["unit_id"] == uid) | (df["legacy_ref_id"].fillna("") == uid)]
            if not hit.empty:
                return {"matches": hit.head(top_k), "relaxed_filters": []}
            relaxed_filters.append(f"unit_id={uid}")

        if slots["pv_code"]:
            hit_pv = df[df["prop_name"].str.contains(slots["pv_code"], case=False, na=False)]
            if not hit_pv.empty:
                df = hit_pv
            else:
                relaxed_filters.append(f"pv_code={slots['pv_code']}")

        if slots["area"]:
            hit_area = df[df["area_name"].str.lower() == slots["area"].lower()]
            if not hit_area.empty:
                df = hit_area
            else:
                relaxed_filters.append(f"area={slots['area']}")

        if slots["max_budget_rm"]:
            hit_b = df[df["rent_rm"] <= slots["max_budget_rm"]]
            if not hit_b.empty:
                df = hit_b
            else:
                relaxed_filters.append(f"max_budget_rm<={slots['max_budget_rm']}")

        if slots["rooms"]:
            hit_r = df[df["rooms"] == slots["rooms"]]
            if not hit_r.empty:
                df = hit_r
            else:
                relaxed_filters.append(f"rooms={slots['rooms']}")

        if slots["furnished"]:
            hit_f = df[df["furnished"].str.lower() == slots["furnished"].lower()]
            if not hit_f.empty:
                df = hit_f
            else:
                relaxed_filters.append(f"furnished={slots['furnished']}")

        if slots["min_parking"]:
            p_num = df["parking_info"].str.extract(r"^(\d+)").astype(float)[0]
            hit_p = df[p_num >= slots["min_parking"]]
            if not hit_p.empty:
                df = hit_p
            else:
                relaxed_filters.append(f"min_parking>={slots['min_parking']}")

        if slots["near_lrt"]:
            hit_lrt = df[df["additional_facilities"].fillna("").str.contains("Near KTM/LRT", case=False)]
            if not hit_lrt.empty:
                df = hit_lrt
            else:
                relaxed_filters.append("near_lrt=True")

        if slots["cooking_allowed"]:
            hit_ck = df[df["additional_facilities"].fillna("").str.contains("Cooking Allowed", case=False)]
            if not hit_ck.empty:
                df = hit_ck
            else:
                relaxed_filters.append("cooking_allowed=True")

        return {
            "matches": df.sort_values("rent_rm", ascending=True).head(top_k),
            "relaxed_filters": relaxed_filters,
        }

    def add_human_slang(self, slang_term: str, normalized_text: str, category="rental_jargon",
                        match_type="phrase", priority=2, target_slot="general"):
        """
        Human-in-the-loop learning: appends new slang safely with basic guardrails
        and reloads the in-memory dictionary immediately.
        """
        slang_clean = slang_term.strip().lower()
        if len(slang_clean) < 2:
            raise ValueError("Guardrail: Single-character slang blocked to prevent word corruption.")
        df = pd.read_csv(self.slang_csv_path)
        if (df["slang_term"].str.lower() == slang_clean).any():
            df.loc[df["slang_term"].str.lower() == slang_clean, ["normalized_text", "source"]] = [
                normalized_text.strip(), "human_learned"
            ]
        else:
            new_row = pd.DataFrame([{
                "slang_term": slang_clean,
                "normalized_text": normalized_text.strip(),
                "category": category,
                "match_type": match_type,
                "priority": priority,
                "target_slot": target_slot,
                "source": "human_learned",
            }])
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.slang_csv_path, index=False)
        self.reload_dictionary()
        return True
