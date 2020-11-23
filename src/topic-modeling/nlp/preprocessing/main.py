from preprocessing import *

if __name__ == "__main__":
    
    original_documents = {
        "title": open("output/title.csv", "w+"),
        "description": open("output/description.csv", "w+"),
        "text": open("output/text.csv", "w+")
    }

    output_tokens = {
        "title": open("output/tokens_title.csv", "w+"),
        "description": open("output/tokens_description.csv", "w+"),
        "text": open("output/tokens_text.csv", "w+")
    }

    output_stop_words = {
        "title": open("output/stop_words_title.csv", "w+"),
        "description": open("output/stop_words_description.csv", "w+"),
        "text": open("output/stop_words_text.csv", "w+")
    }

    output_lemmas = {
        "title": open("output/lemmas_title.csv", "w+"),
        "description": open("output/lemmas_description.csv", "w+"),
        "text": open("output/lemmas_text.csv", "w+")
    }

    document_set = get_set("news.csv")
    document_features = document_set.columns

    titles = []
    texts = []
    descriptions = []

    for feat in document_features:

        for doc in document_set[feat].items():

            # Writes the original document just for comparing
            print(doc[1], file=original_documents[feat])

            # Extracts tokens and write in a feature specific file
            tokens = tokenize(doc[1])
            print(tokens, file=output_tokens[feat])

            # Removes stop word and write in a feature specific file
            no_stop_words = drop_stop_words(tokens)
            print(no_stop_words, file=output_stop_words[feat])

            lemmas = lemmatize(no_stop_words)
            print(lemmas, file=output_lemmas[feat])

    for output_file in original_documents.values(): output_file.close()
    for output_file in output_tokens.values(): output_file.close()
    for output_file in output_stop_words.values(): output_file.close()
    for output_file in output_lemmas.values(): output_file.close()
