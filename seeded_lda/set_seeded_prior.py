from text_preprocessing import get_seed_term_candidates


def set_seeded_prior(mdl, seed_lexicon, topic_name_to_id, seed_weight=4.0, regular_weight=0.001):
    seeded_word_count = 0
    missing_seed_words = []

    for topic_name, words in seed_lexicon.items():
        topic_id = topic_name_to_id[topic_name]
        
        for word in words:
            priors = [regular_weight] * mdl.k
            priors[topic_id] = seed_weight

            anchored = False
            for candidate in get_seed_term_candidates(word):
                try:
                    mdl.set_word_prior(candidate, priors)
                    seeded_word_count += 1
                    anchored = True
                    break
                except ValueError:
                    continue

            if not anchored:
                missing_seed_words.append(word)

    print(f"Successfully anchored {seeded_word_count} seed words.")
    if missing_seed_words:
        print(f"Warning: The following seed words were not found in the model's vocabulary and were skipped:\n{', '.join(missing_seed_words)}")

    return mdl 