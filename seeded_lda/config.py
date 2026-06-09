seed_lexicon = {
    "propaganda": ["propaganda", "disinformation", "misinformation", "influence_operation"], # "spread_lie", "influence_operation", "influence_campaign", "spread_propaganda", "misinform", "disinformation_campaign", "information_warfare", "information_operation"],
    "election_interference": ["election_interference",   "election_meddling", "vote_manipulation", "influence_election", "voter_fraud"], # "foreign_interference", 'interference',"meddling","influence_elections", "election_influence", "voter_fraud", "election_integrity", "rig_election"],
    "hacking": ["hack", "cyber_attack", "cyberattack", "cyber_espionage", "breach"], # "hacking", "hacker"
    "social_media": ["social_media", "facebook", "twitter", "instagram", "tiktok", "reddit", "youtube", "internet", "whatsapp", "telegram", ], # "app", "social_network", "telegram_channel","social_platform", "tweet", "influencer"],
    "artificial_intelligence": ["artificial_intelligence", "ai", "generative_ai"], # "ai_generate", "ai_bot", "generative_ai", "chatbot", "large_language_model", "deepfake"],
    "us_elections": [ "presidential_election", "trump", "biden", "harris", "democrats", "republicans", "democratic_party", "republican_party"],
    "russia": ["russia", "putin", "kremlin", "moscow"], #, "ira", "internet_research_agency", "russian_government", "russian_military", "russian", "soviet", "kgb", "wagner_group"], 
    "china": ["china", "xi_jinping", "beijing", "ccp"], # "chinese","xi_jinping", "beijing", "chinese_government", "chinese_military", "ccp", "chinese_communist", "chinese_economy"],
    "iran": ["iran", "ayatollah", "tehran", "khamenei"], # "iranian", "rouhani", "tehran", "iranian_government", "iranian_military", "ayatollah", "iranian_revolutionary_guard", "khamenei", "irgc" ],
}


election_dates = {
    "US Election": "2024-11-05",
}

custom_words_to_remove = [
    # 1. REPORTING VERBS (The mechanics of journalism)
    "say", "tell", "report", "state", "add", "continue", "note", "speak",
    "announce", "publish", "write", "claim", "ask", "answer", "respond",
    "explain", "describe", "mention", "declare", "comment", "discuss", "happen",
      "occur", "emerge", "reveal", "confirm", "deny", "allege", "suggest", "indicate", 
      "highlight", "point_out", "underscore", "emphasize", "assert", "acknowledge", 
      "admit", "refute", "clarify", "illustrate", "demonstrate", "reiterate", 
      "stress", "note_that", "observe", "report_that", "state_that", "claim_that", 
      "suggest_that", "indicate_that", "highlight_that", "point_out_that", "underscore_that", 
      "address", "mention",  "welcome", "criticize", "applaud", "denounce",
      "allege", "confirm", "deny", "claim", "suggest", "indicate", "highlight", "point_out", "underscore", "emphasize", "assert",
    
    # 2. JOURNALISM & PUBLISHING JARGON
    "news", "reporter", "correspondent", "press", "release", "statement", 
    "source", "interview", "article", "column", "editorial", "editor", 
    "broadcast", "medium", "newspaper", "magazine", "coverage", "story",
    "headline", "journalist", "author",
    
    # 3. WEB & UI ARTIFACTS (Scraping noise)
    "read", "subscribe", "newsletter", "advertisement", "click", "share", 
    "update", "link", 
    "page", "home", "copyright", "follow", "loading", "sign", "register",
    "cookie", "term", "condition", "browser",
    "download", "menu", "search", "advertisement", "ad", "continue_ad", "ad_free",
    
    # 4. TIME & DATE MARKERS (Usually irrelevant to the actual topic)
    "today", "yesterday", "tomorrow", "week", "month", "year", "time", 
    "day", "minute", "hour", "morning", "afternoon", "evening", "daily", 
    "weekly", "annual", "monday", "tuesday", "wednesday", "thursday", 
    "friday", "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november", "december",
    
    # 5. TITLES, PRONOUNS, & GENERIC FILLER
    "mr", "mrs", "ms", "dr", "sir", "madam", "people", "person", "man", 
    "woman", "like", "know", "think", "look", "good", "new", "old", "way", 
    "thing", "come", "go", "take", "get", "make", "use", "want", "find",
    "need", "work", "give", "try", "leave", "call", "lot", "bit", "get", "want", "say", "see", "look", "meet",

    "dot", "com", "cn", "www", "http", "https", "href", "html"
]