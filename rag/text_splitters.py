def recursive_character_text_splitter(text, chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex
    ).split_text(text)


def chunk_text(text, chunk_size=1000, overlap=100):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]


def spacy_chunk(text, max_words=200):
    import spacy

    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in doc.sents:
        words = sent.text.split()
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_len = len(words)
        else:
            current_chunk.extend(words)
            current_len += len(words)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks