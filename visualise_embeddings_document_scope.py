import codecs
from functools import reduce
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import glob
import string
from gensim.models import KeyedVectors
import re
import numpy as np
import nltk


with open('stopwords.txt', encoding='utf-8') as f:
    STOPWORDS = f.read().splitlines()
    
def load_files(path):
    docs = []
    files = glob.glob(path+'/**/*', recursive=True)
    for file in files:
        try:
            with open(file, encoding='utf-8') as f:
                docs.extend(f.read().splitlines())
        except:
            pass
    return docs


def remove_entity(sentence):
    return re.sub('\<(.*?)\>', '', sentence)


def list_people(docs):
    ## List all people appearing in documents
    all_people = []
    added_people = []
    for doc in docs:
        entities = re.findall('\<Entity(.*?)\>', doc)
        for item in entities:
            name = re.findall('name="(.*?)"', item)[0]
            if name not in added_people:
                added_people.append(name)
                try:
                    person = {
                        'name': re.findall('name="(.*?)"', item)[0],
                        'type': re.findall('type="(.*?)"', item)[0],
                        'category': re.findall('category="(.*?)"', item)[0]
                    }
                    all_people.append(person)
                except:
                    pass
    return all_people


def extract_sentences(doc, tokenizer):
    ## You might need to execute
    ## nltk.download('punkt') before using nltk
    return(tokenizer.tokenize(doc))


def find_sent_with_person(p_name, all_sentences):
    selected_sentences = []
    for doc in all_sentences:
        for sentence in doc:
            if p_name in sentence:
                selected_sentences.append(sentence)
    return selected_sentences

def find_sent_with_person_within_document(doc, tokenizer):
    selected_sentences = []
    for doc in all_sentences:
        doc_sentences = []
        for sentence in doc:
            if p_name in sentence:
                doc_sentences.append(sentence)
        selected_sentences.append(sentence)
    return selected_sentences

## files needed for tensorflow projector


def save_model(model):
    with open('word2vec_emb.tsv', 'w', encoding='utf-8') as vec_file, open('word2vec_meta.tsv', 'w', encoding='utf-8') as metafile:
        for word in list(model.wv.vocab):
            vec = '\t'.join(map(str, model[word]))
            vec_file.write(vec+'\n')
            metafile.write(word+'\n')


def extract_neighbor_words(sentences, entity, words_before=3, words_after=3, stopwords=STOPWORDS, keep_person=False):
    extracted_words = []
    for sentence in sentences:
        persons = re.findall(f'<Entity name="{entity}".*?">(.*?)</', sentence)

        ## Remove entity
        ## <Entity name="Tomasz Sekielski" type="person" category="dziennikarze">Tomasz Sekielski</Entity>
        ## will result as Tomasz Sekielski
        sentence = remove_entity(sentence)

        ## Remove digits
        sentence = sentence.translate(str.maketrans('', '', string.digits))

        ## Remove punctuation marks
        sentence = sentence.translate(
            str.maketrans('', '', string.punctuation))

        ## Remove stopwords
        sentence = [word for word in sentence.split(
        ) if word not in stopwords and len(word) > 1]

        before_and_afters = []
        for person in persons:
            # print(sentence)
            try:
                before = sentence[:sentence.index(person.split()[0])]
                before = before[-words_before:]
                after = sentence[sentence.index(person.split()[-1]) + 1:]
            except Exception as e:
                # print(f""Sentence {sentence} does not contain '{person.split()[0]}'")
                continue

            sentence = after  # update sentence to get different mention next time

            after = after[:words_after]

            if keep_person:
                words_before_after = before + person + after
            else:
                words_before_after = before + after

            ## Lemmatisation (not obligatory)
            ## Sometimes returns many different results for specific words.
            ## For exmaple for zamek returns zamek:s1, zamek:s2
            ## Try: morf.analyse('zamki')
            ##      morf.analyse('zamki')[0][2][1].split(':', 1)[0]
            # words_before_after = [extract_lemm(morf.analyse(word)) for word in words_before_after]
            ## in case word2vec cares about capital letters
            words_before_after = [word.lower() for word in words_before_after]
            extracted_words.append(words_before_after)

    return extracted_words

def extract_neighbor_words_doc_scope(list_of_sentences, entity, words_before=3, words_after=3, stopwords=STOPWORDS, keep_person=False):
    extracted_words_corpus = []
    for sentences in list_of_sentences:
        extracted_words = []
        for sentence in sentences:
            persons = re.findall(f'<Entity name="{entity}".*?">(.*?)</', sentence)

            ## Remove entity
            ## <Entity name="Tomasz Sekielski" type="person" category="dziennikarze">Tomasz Sekielski</Entity>
            ## will result as Tomasz Sekielski
            sentence = remove_entity(sentence)

            ## Remove digits
            sentence = sentence.translate(str.maketrans('', '', string.digits))

            ## Remove punctuation marks
            sentence = sentence.translate(
                str.maketrans('', '', string.punctuation))

            ## Remove stopwords
            sentence = [word for word in sentence.split(
            ) if word not in stopwords and len(word) > 1]

            before_and_afters = []
            for person in persons:
                # print(sentence)
                try:
                    before = sentence[:sentence.index(person.split()[0])]
                    before = before[-words_before:]
                    after = sentence[sentence.index(person.split()[-1]) + 1:]
                except Exception as e:
                    # print(f""Sentence {sentence} does not contain '{person.split()[0]}'")
                    continue

                sentence = after  # update sentence to get different mention next time

                after = after[:words_after]

                if keep_person:
                    words_before_after = before + person + after
                else:
                    words_before_after = before + after

                ## Lemmatisation (not obligatory)
                ## Sometimes returns many different results for specific words.
                ## For exmaple for zamek returns zamek:s1, zamek:s2
                ## Try: morf.analyse('zamki')
                ##      morf.analyse('zamki')[0][2][1].split(':', 1)[0]
                # words_before_after = [extract_lemm(morf.analyse(word)) for word in words_before_after]
                ## in case word2vec cares about capital letters
                words_before_after = [word.lower() for word in words_before_after]
                extracted_words.append(words_before_after)
        extracted_words_corpus.append( [item for sublist in extracted_words for item in sublist])

    return extracted_words_corpus


def save_for_tensorprojector(vec, meta, filename="tensorvectors"):
    def glue(acc, el):
        return "{}\t{}".format(acc, el)

    with codecs.open("{}.tsvec".format(filename), 'w', "utf-8") as f:
        for v in vec:
            f.write(reduce(glue, v)+'\n')

    with codecs.open("{}.meta".format(filename), 'w', "utf-8") as f:
        for w in meta:
            #words, person = w
            #line = "{}\t{}\n".format(words, person)
            line = "{}\n".format(w)
            f.write(line)



if __name__ == "__main__":
    PATH = './categorization/learningData'


    word2vec = KeyedVectors.load("./word2vec/word2vec_100_3_polish.bin")
    tokenizer = nltk.data.load('polish.pickle')

    docs = load_files(PATH)

    ## list all people marked in text
    ## returns list of dicts, each person has attr: name, category, type
    people = list_people(docs)

    ## returns list of sentences in each document
    sentences = [extract_sentences(document, tokenizer) for document in docs]

    vector_means = []
    metadata = []
    ## find sentences for specific person
    for person in people:
        # print(f"Processing {person['name']}")
        filtered_sentences = find_sent_with_person_within_document(person['name'], sentences)

        extracted_neighbour_groups = extract_neighbor_words(
            filtered_sentences, person['name'], words_before=1, words_after=2)

        for group in extracted_neighbour_groups:
            vectors = []
            for word in group:
                try:
                    vectors.append(word2vec[word])
                except Exception as e:
                    continue
                    #print(f"{word} not found")
            if len(vectors) > 0:
                #metadata.append((group, person['name']))
                metadata.append(person['name'])
                vector_means.append(np.add.reduce(vectors)/len(vectors))


save_for_tensorprojector(vector_means, metadata)


# obecnie nam robi jako one mention scope


