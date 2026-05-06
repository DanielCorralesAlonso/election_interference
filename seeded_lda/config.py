seed_lexicon = {
    "propaganda": ["propaganda", "disinformation", "misinformation", "spread_lie", "fake_news", "influence_operation", "influence", "influence_campaign", "spread_propaganda", "misinform", "disinform", "manipulate", "manipulated", "manipulation", "fake_account", "disinformation_campaign"],
    "election_interference": ["election_interference", "foreign_interference", 'interference', "election_meddling", "vote_manipulation", "ballot_interference", "influence_election", "influence_elections", "election_influence"],
    "hacking": ["hack", "hacking", "cyber_attack", "cyberattack", "hacker", "fake_site", "data_breach", "malware", "phishing", "ransomware",],
    "social_media": ["social_media", "app", "social_network", "social_medium", "facebook", "twitter", "instagram", "tiktok", "reddit", "youtube", "internet", "whatsapp", "telegram", "telegram_channel","social_platform"],
    "artificial_intelligence": ["artificial_intelligence", "ai", "gen_ai", "ai_generate", "ai_generation", "ai_bot", "ai_bots", "generative_ai", "chatbots", "ai_trolls", "ai_based", "a.i."],
    "security": ["threat", "security", "security_concern", "cybersecurity", "vulnerability", "risk", "defense", "countermeasure", "mitigation", "national_security", "u.s._intelligence"],
    "us_elections": ["us_elections", "u.s._elections", "presidential_election", "nov._election", "u.s._midterm", "donald_trump", "joe_biden", "gop", "kamala_harris", "democrats", "republicans", "democratic_party", "republican_party", "midterm_elections", "presidential_elections", "election_security"],
    "russia": ["russia", "putin", "kremlin", "moscow", "ira", "internet_research_agency", "russian_government", "russian_military", "russian"],
    "china": ["china", "chinese","xi_jinping", "beijing", "chinese_government", "chinese_military", "ccp", "chinese_communist"],
    "iran": ["iran", "irarian", "rouhani", "tehran", "iranian_government", "iranian_military", "ayatollah", "iranian_revolutionary_guard"],
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