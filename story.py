story_subject = ['hiking', 'dog', 'football', 'motorcycle', 'dogs','snowy', 'little', 'climbing']

def split_subject(caption):
    subject = caption.split()
    for i in subject:
        if i in story_subject:
            file_path = 'sentences/' + i + '.txt'
            return file_path
    return None  # Return None if no subject is found

def read_sentences(file_path):
    with open(file_path, 'r') as file:
        sentences = file.read().split('\n')
        return {sentence.strip() for sentence in sentences if sentence.strip()}

def generate_story(sentences, caption):
    for _ in range(5):
        sentence = next((sentence for sentence in sentences), None)
        if sentence:
            caption += ' ' + sentence
            sentences.remove(sentence)
        else:
            break
    return caption

def generate_story_from_caption(caption):
    file_path = split_subject(caption)
    if file_path:
        sentences = read_sentences(file_path)
        caption = caption[0].upper() + caption[1:]
        caption += '.'
        return generate_story(sentences, caption)
    else:
        caption = caption[0].upper() + caption[1:]
        caption += '.'
        return caption  # Return the caption only if no subject is found
