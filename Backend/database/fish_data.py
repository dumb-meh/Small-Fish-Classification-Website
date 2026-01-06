# Static fish dataset for FishAI
# Keys are lowercase transliterations of the common name

FISH_DATA = {
    "bele": {
        "name_bn": "বেলে",
        "name_en": "Bele",
        "scientific_name": "Glossogobius giuris",
        "description": "A slender, predatory goby with a large mouth and two dorsal fins. Often found near the bottom.",
        "size": "15-25 cm",
        "primary_rivers": "Padma, Jamuna, Meghna",
        "river_habitat": "Sandy, muddy bottoms",
        "diet": "Carnivore",
        "culinary_note": "Not a major food fish, sometimes used in curries or as bait."
    },
    "chela": {
        "name_bn": "চেলা",
        "name_en": "Chela",
        "scientific_name": "Chela cachius",
        "description": "A small, silvery fish with a deeply forked tail and a sharp keel on its belly. Very active and swift.",
        "size": "5-8 cm",
        "primary_rivers": "Rivers & Haors nationwide",
        "river_habitat": "Surface waters, streams",
        "diet": "Omnivore (plankton)",
        "culinary_note": "Often dried or fried whole; used in small fish preparations."
    },
    "guchi": {
        "name_bn": "গুঁচি",
        "name_en": "Guchi",
        "scientific_name": "Mystus vittatus",
        "description": "A small catfish with long barbels, a forked tail, and a distinctive dark stripe along its side.",
        "size": "10-15 cm",
        "primary_rivers": "Padma, Chalan Beel",
        "river_habitat": "Weedy, muddy areas",
        "diet": "Omnivore",
        "culinary_note": "A tasty small catfish, popular in spicy curries and fried preparations."
    },
    "kachki": {
        "name_bn": "কাচকি",
        "name_en": "Kachki",
        "scientific_name": "Corica soborna",
        "description": "A tiny, almost transparent fish, one of the smallest in Bangladesh. Very delicate.",
        "size": "2-4 cm",
        "primary_rivers": "Padma, Meghna estuaries",
        "river_habitat": "Open water, estuaries",
        "diet": "Planktivore",
        "culinary_note": "Highly prized. Made into \"Kachki Jhaal\" (spicy curry) or mixed into batter for fritters."
    },
    "kata phasa": {
        "name_bn": "কাঁটা ফাঁসা",
        "name_en": "Kata Phasa",
        "scientific_name": "Setipinna phasa",
        "description": "A small, silvery fish from the anchovy family, known for its sharp belly scutes (spines).",
        "size": "8-12 cm",
        "primary_rivers": "Coastal rivers, Sundarbans",
        "river_habitat": "Estuarine, brackish",
        "diet": "Planktivore",
        "culinary_note": "Often dried or cooked in a sharp, mustard-based curry similar to Hilsa."
    },
    "mola": {
        "name_bn": "মলা",
        "name_en": "Mola",
        "scientific_name": "Amblypharyngodon mola",
        "description": "A small, deep-bodied, silvery fish with a small head. Feeds on algae and detritus.",
        "size": "5-10 cm",
        "primary_rivers": "Ponds, beels, rivers nationwide",
        "river_habitat": "Weedy, shallow waters",
        "diet": "Herbivore (algae)",
        "culinary_note": "Eaten whole (fried) and is an excellent source of Vitamin A and calcium."
    },
    "nama chanda": {
        "name_bn": "নামা চান্দা",
        "name_en": "Nama Chanda",
        "scientific_name": "Chanda nama",
        "description": "A very thin, almost diamond-shaped, translucent fish. Also called the \"Elongate Glassy Perchlet.\"",
        "size": "5-8 cm",
        "primary_rivers": "Slow-moving rivers, ponds",
        "river_habitat": "Still, clear water",
        "diet": "Carnivore (small insects)",
        "culinary_note": "Usually dried or fried into crisp chips."
    },
    "pabda": {
        "name_bn": "পাবদা",
        "name_en": "Pabda",
        "scientific_name": "Ompok pabda",
        "description": "A smooth, scaleless catfish with a broad, flat head and long barbels. Highly valued.",
        "size": "15-25 cm",
        "primary_rivers": "Padma, Jamuna",
        "river_habitat": "Deep river channels",
        "diet": "Carnivore",
        "culinary_note": "A celebrated delicacy, famous for its soft, flavorful flesh. Cooked in light curries (\"Pabda Jhol\") or steamed."
    },
    "puti": {
        "name_bn": "পুঁটি",
        "name_en": "Puti",
        "scientific_name": "Puntius sophore",
        "description": "A common, small, silvery-bronze fish with a distinct black spot near the tail base.",
        "size": "5-12 cm",
        "primary_rivers": "Rivers & floodplains nationwide",
        "river_habitat": "Ponds, canals, beels",
        "diet": "Omnivore",
        "culinary_note": "Extremely popular. Fried whole, cooked in chorchori (mixed veg stew), or made into dried shutki."
    },
    "tengra": {
        "name_bn": "টেংরা",
        "name_en": "Tengra",
        "scientific_name": "Mystus tengara",
        "description": "A small to medium catfish with spotted fins, long barbels, and a strong dorsal spine.",
        "size": "12-18 cm",
        "primary_rivers": "Padma, Chalan Beel",
        "river_habitat": "Muddy riverbeds",
        "diet": "Omnivore",
        "culinary_note": "Famous for its Tengra Jhaal – a hot and sour curry, a beloved dish in Bengali cuisine."
    }
}


def get_fish_data(key: str):
    """Return fish data for a given key (case-insensitive)."""
    if not key:
        return None
    key = key.strip().lower()
    # Some keys might have alternative spacing - normalize
    if key in FISH_DATA:
        return FISH_DATA[key]
    # Try replacing spaces with single space normalizations
    alt = ' '.join(key.split())
    return FISH_DATA.get(alt)
