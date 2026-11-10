"""
backend/src/normalization.py
----------------------------
Ingredient Normalization Engine for the NOSH Conversational Recipe System.

Purpose:
  1. Standardizes raw ingredient strings into clean canonical culinary keys.
  2. Strips quantities, units of measurement, preparation methods, and noise.
  3. Lemmatizes plurals (e.g. 'tomatoes' -> 'tomato', 'onions' -> 'onion').
  4. Resolves common culinary synonyms (e.g. 'scallion' -> 'green onion').
  5. Maps ingredients to allergen and dietary restriction categories (e.g. 'dairy', 'peanuts', 'gluten').
  6. Preserves critical culinary distinctions (e.g. 'tomato paste' vs 'fresh tomato').

Note:
  This module does NOT alter the original user-facing recipe text. It creates
  normalized indexing tags for fast Inverted Index lookups and exact constraint satisfaction.
"""

import re
from typing import Set, List, Dict, Tuple

# =============================================================================
# 1. Measurement Units to Strip
# =============================================================================
# Standard cooking measurements, packaging terms, and container descriptors.
MEASUREMENT_UNITS: Set[str] = {
    # Volume
    "cup", "cups", "c",
    "tablespoon", "tablespoons", "tbsp", "tbsps", "tbs", "tb",
    "teaspoon", "teaspoons", "tsp", "tsps",
    "fluid", "fl", "fl oz", "pint", "pints", "pt",
    "quart", "quarts", "qt", "gallon", "gallons", "gal",
    "milliliter", "milliliters", "ml", "liter", "liters", "l",
    
    # Weight
    "ounce", "ounces", "oz",
    "pound", "pounds", "lb", "lbs",
    "gram", "grams", "g", "kilogram", "kilograms", "kg",
    
    # Containers & Shapes
    "can", "cans", "tin", "tins", "jar", "jars", "bottle", "bottles",
    "package", "packages", "pkg", "pkgs", "packet", "packets",
    "bag", "bags", "box", "boxes", "carton", "cartons",
    "slice", "slices", "piece", "pieces", "cube", "cubes",
    "pinch", "pinches", "dash", "dashes", "drop", "drops",
    "clove", "cloves", "head", "heads", "bunch", "bunches",
    "sprig", "sprigs", "stalk", "stalks", "stick", "sticks",
    "strip", "strips", "loaf", "loaves", "sheet", "sheets",
    "envelope", "envelopes", "container", "containers", "batch"
}

# =============================================================================
# 2. Preparation Adjectives & Cooking Methods to Strip
# =============================================================================
# Culinary preparation words that describe HOW an ingredient is cut or handled,
# but do not change the core ingredient identity.
PREPARATION_TERMS: Set[str] = {
    # Cutting methods
    "chopped", "diced", "minced", "sliced", "grated", "shredded",
    "crushed", "julienned", "mashed", "pureed", "cubed", "slivered",
    "halved", "quartered", "torn", "cracked", "flaked", "ground",
    
    # Processing states
    "peeled", "unpeeled", "seeded", "deseeded", "pitted", "stemmed",
    "trimmed", "skinned", "skinless", "boneless", "bone-in",
    "drained", "rinsed", "sifted", "strained", "beaten", "whisked",
    "melted", "softened", "chilled", "frozen", "thawed",
    "toasted", "roasted", "boiled", "baked", "fried", "sauteed",
    "cooked", "uncooked", "raw", "steamed", "caramelized",
    
    # Degrees and descriptors
    "finely", "coarsely", "thinly", "thickly", "lightly", "gently",
    "roughly", "freshly", "warm", "cold", "hot", "room", "temperature",
    "divided", "optional", "extra", "plus", "more", "needed", "taste",
    "packed", "heaping", "level", "scant", "dry", "dried", "fresh"
}

# Modifiers to ignore when extracting core root nouns (e.g. "large yellow onion" -> "onion")
SIZE_AND_QUALITY_MODIFIERS: Set[str] = {
    "small", "medium", "large", "extra-large", "jumbo",
    "lean", "extra-lean", "fat-free", "low-fat", "reduced-fat",
    "organic", "all-natural", "homemade", "store-bought",
    "good", "quality", "ripe", "overripe"
}

# =============================================================================
# 3. Synonym & Culinary Equivalence Mapping
# =============================================================================
# Normalizes regional vocabulary to a canonical standard term.
SYNONYM_MAP: Dict[str, str] = {
    "scallion": "green onion",
    "scallions": "green onion",
    "spring onion": "green onion",
    "spring onions": "green onion",
    "cilantro": "coriander",
    "capsicum": "bell pepper",
    "sweet pepper": "bell pepper",
    "sweet peppers": "bell pepper",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "garbanzo bean": "chickpea",
    "garbanzo beans": "chickpea",
    "garbanzos": "chickpea",
    "chickpeas": "chickpea",
    "confectioners sugar": "powdered sugar",
    "icing sugar": "powdered sugar",
    "caster sugar": "granulated sugar",
    "white sugar": "sugar",
    "granulated white sugar": "sugar",
    "all purpose white flour": "all purpose flour",
    "plain flour": "all purpose flour",
    "cremini mushroom": "baby bella mushroom",
    "portobello mushroom": "portobello"
}

# =============================================================================
# 4. Allergen & Dietary Exclusion Taxonomy
# =============================================================================
# Used for deterministic constraint enforcement in Task 2 & Task 3.
# When a user says "I cannot have dairy", any ingredient matching this set is excluded.
ALLERGEN_TAXONOMY: Dict[str, Set[str]] = {
    "dairy": {
        "milk", "butter", "cheese", "cream", "heavy cream", "sour cream",
        "half-and-half", "yogurt", "buttermilk", "ghee", "parmesan",
        "cheddar", "mozzarella", "ricotta", "mascarpone", "feta", "brie",
        "gouda", "provolone", "swiss cheese", "whey", "casein", "curd",
        "cream cheese", "condensed milk", "evaporated milk"
    },
    "peanuts": {
        "peanut", "peanuts", "peanut butter", "peanut oil", "peanut flour"
    },
    "tree_nuts": {
        "almond", "almonds", "walnut", "walnuts", "cashew", "cashews",
        "pecan", "pecans", "pistachio", "pistachios", "hazelnut", "hazelnuts",
        "macadamia", "macadamias", "pine nut", "pine nuts", "almond flour",
        "almond milk", "almond meal", "almond butter"
    },
    "gluten": {
        "flour", "all purpose flour", "wheat flour", "bread flour", "wheat",
        "barley", "rye", "semolina", "spelt", "bulgur", "couscous", "pasta",
        "spaghetti", "penne", "macaroni", "fettuccine", "noodle", "noodles",
        "breadcrumbs", "panko", "soy sauce", "vital wheat gluten"
    },
    "egg": {
        "egg", "eggs", "egg yolk", "egg yolks", "egg white", "egg whites",
        "mayonnaise"
    },
    "soy": {
        "soy", "soya", "soy sauce", "tofu", "edamame", "miso", "tempeh",
        "soy milk", "soy protein"
    },
    "shellfish": {
        "shrimp", "shrimps", "crab", "crabs", "lobster", "lobsters", "prawn",
        "prawns", "clam", "clams", "mussel", "mussels", "oyster", "oysters",
        "scallop", "scallops", "crawfish", "crayfish"
    },
    "fish": {
        "salmon", "tuna", "cod", "tilapia", "trout", "halibut", "anchovy",
        "anchovies", "sardine", "sardines", "bass", "snapper", "fish sauce"
    },
    "pork": {
        "bacon", "pork", "ham", "prosciutto", "pancetta", "pork chop",
        "pork loin", "sausage", "chorizo", "lard", "pepperoni"
    },
    "beef": {
        "beef", "ground beef", "steak", "roast beef", "beef broth",
        "beef stock", "sirloin", "brisket", "veal"
    }
}

# Inverted mapping: ingredient keyword -> set of allergen categories
_KEYWORD_TO_ALLERGENS: Dict[str, Set[str]] = {}
for category, keywords in ALLERGEN_TAXONOMY.items():
    for kw in keywords:
        if kw not in _KEYWORD_TO_ALLERGENS:
            _KEYWORD_TO_ALLERGENS[kw] = set()
        _KEYWORD_TO_ALLERGENS[kw].add(category)


# =============================================================================
# 5. Core Normalization Logic
# =============================================================================

def lemmatize_word(word: str) -> str:
    """
    Rule-based lemmatizer to convert common plural nouns to singular form.
    Guarantees fast, deterministic results without external heavy NLP packages.
    """
    w = word.lower()
    
    # Exceptions that should not be stripped
    if w in {"cheese", "molasses", "couscous", "hummus", "bass", "swiss", "citrus"}:
        return w
        
    # Words ending in 'ies' -> 'y' (e.g. berries -> berry, cherries -> cherry)
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
        
    # Words ending in 'ves' -> 'f' or 'fe' (e.g. halves -> half, leaves -> leaf)
    if w.endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
        
    # Words ending in 'es' -> '' (e.g. tomatoes -> tomato, potatoes -> potato, dishes -> dish)
    if w.endswith("es") and len(w) > 4 and not w.endswith("cheese"):
        # preserve words like 'olives' -> 'olive'
        if w.endswith("ves"):
            return w[:-1]
        return w[:-2]
        
    # Standard plural 's'
    if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        return w[:-1]
        
    return w


def clean_raw_ingredient(raw_str: str) -> str:
    """
    Takes a raw scraped ingredient line and extracts the core ingredient phrase:
    - Removes numbers, fractions (e.g. 1/2, 3/4), and parenthetical weights (e.g. (10 oz))
    - Removes measurement units and preparation techniques
    - Lemmatizes plural nouns to singular form
    - Normalizes synonyms
    """
    if not raw_str or not isinstance(raw_str, str):
        return ""
        
    text = raw_str.lower()
    
    # Remove parenthetical comments like "(10.75 ounce)", "(optional)", "(about 2 cups)"
    text = re.sub(r"\(.*?\)", " ", text)
    
    # Remove numbers, fractions, punctuation except hyphens
    text = re.sub(r"[\d/\\,\.\*\+\;\:\!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    words = text.split()
    
    # Filter out units of measurement, preparation terms, and size modifiers
    meaningful_words = []
    for w in words:
        w_clean = w.strip(" -")
        if not w_clean or len(w_clean) < 2:
            continue
            
        w_lemma = lemmatize_word(w_clean)
        
        # Check if the word is a measurement unit or prep term
        if w_clean in MEASUREMENT_UNITS or w_lemma in MEASUREMENT_UNITS:
            continue
        if w_clean in PREPARATION_TERMS or w_lemma in PREPARATION_TERMS:
            continue
        if w_clean in SIZE_AND_QUALITY_MODIFIERS:
            continue
            
        meaningful_words.append(w_lemma)
        
    if not meaningful_words:
        return ""
        
    joined = " ".join(meaningful_words)
    
    # Apply synonym mapping
    return SYNONYM_MAP.get(joined, joined)


def extract_normalized_keys(raw_ingredient: str) -> Set[str]:
    """
    Extracts a set of searchable keys from a raw ingredient string.
    Example:
      Input:  "2 large yellow onions, finely diced"
      Output: {"yellow onion", "onion"}
      
      Input:  "1 cup grated parmesan cheese"
      Output: {"parmesan cheese", "parmesan", "cheese"}
    """
    primary = clean_raw_ingredient(raw_ingredient)
    if not primary:
        return set()
        
    keys: Set[str] = {primary}
    
    # If multi-word (e.g. "yellow onion" or "chicken breast"), also extract the core noun
    parts = primary.split()
    if len(parts) > 1:
        # The last word is usually the core culinary noun (e.g. "onion", "cheese", "soup")
        last_word = parts[-1]
        # Avoid treating plant substitutes (peanut butter, almond milk, coconut cream) as generic dairy butter/milk/cream
        is_plant_sub = any(ps in primary for ps in ["peanut butter", "almond butter", "cashew butter", "apple butter", "cocoa butter", "almond milk", "soy milk", "coconut milk", "oat milk", "coconut cream"])
        if len(last_word) > 2 and last_word not in {"oil", "powder", "paste", "sauce"}:
            if not (last_word in {"butter", "milk", "cream"} and is_plant_sub):
                keys.add(last_word)
            
        # Common compound nouns
        if "cheese" in parts and len(parts) >= 2:
            keys.add("cheese")
            keys.add(parts[0]) # e.g. "parmesan"
        if "onion" in parts:
            keys.add("onion")
        if "pepper" in parts:
            keys.add("pepper")
        if "chicken" in parts:
            keys.add("chicken")
        if "beef" in parts:
            keys.add("beef")
        if "potato" in parts:
            keys.add("potato")
        if "tomato" in parts:
            keys.add("tomato")
        if "mushroom" in parts:
            keys.add("mushroom")
            
    return keys


# Plant-based alternatives that contain words like "milk" or "butter" but are NOT dairy
PLANT_DAIRY_EXCEPTIONS: Set[str] = {
    "peanut butter", "almond butter", "cashew butter", "apple butter", "cocoa butter", "sunflower butter",
    "almond milk", "soy milk", "coconut milk", "oat milk", "rice milk", "hemp milk", "cashew milk",
    "coconut cream", "coconut butter", "shea butter"
}

def detect_allergens(keys: Set[str]) -> Set[str]:
    """
    Checks if any of the ingredient keys trigger known dietary/allergen restrictions.
    Returns set of allergen labels (e.g. {"dairy", "gluten"}).
    Accurately avoids false positives (e.g. plant milks and plant butters triggering dairy).
    """
    detected: Set[str] = set()
    
    for key in keys:
        # Check if this ingredient is an explicit plant-based exception to dairy
        is_plant_dairy_sub = any(exc in key for exc in PLANT_DAIRY_EXCEPTIONS)
        
        if key in _KEYWORD_TO_ALLERGENS:
            for a in _KEYWORD_TO_ALLERGENS[key]:
                if a == "dairy" and is_plant_dairy_sub:
                    continue
                detected.add(a)
        else:
            # Substring check for compound terms
            for kw, cats in _KEYWORD_TO_ALLERGENS.items():
                if kw in key:
                    for c in cats:
                        if c == "dairy" and is_plant_dairy_sub:
                            continue
                        detected.add(c)
    return detected
