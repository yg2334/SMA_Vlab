import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
nltk.download('stopwords')
import pandas as pd 
#st.set_page_config(layout="wide")

st.title("KJSIT Text Summarization Virtual Lab")
st.text("LY A1 Batch")
st.write("Definition : Summarization can be defined as a task of producing a concise and fluent summary while preserving key information and overall meaning.")
st.header("Impact")
st.write("Summarization systems often have additional evidence they can utilize in order to specify the most important topics of document(s). For example, when summarizing blogs, there are discussions or comments coming after the blog post that are good sources of information to determine which parts of the blog are critical and interesting.\n In scientific paper summarization, there is a considerable amount of information such as cited papers and conference information which can be leveraged to identify important sentences in the original paper.")
st.header("Types of Text Summarization")
st.subheader("1) Abstractive Summarization:")
st.write("Abstractive methods select words based on semantic understanding, even those words did not appear in the source documents. It aims at producing important material in a new way. They interpret and examine the text using advanced natural language techniques in order to generate a new shorter text that conveys the most critical information from the original text.\nIt can be correlated to the way human reads a text article or blog post and then summarizes in their own word.")
st.write("Input document → understand context → semantics → create own summary.")
st.subheader("2) Extractive Summarization:")
st.write("Extractive methods attempt to summarize articles by selecting a subset of words that retain the most important points.This approach weights the important part of sentences and uses the same to form the summary. Different algorithm and techniques are used to define weights for the sentences and further rank them based on importance and similarity among each other.")
st.write("Input document → sentences similarity → weight sentences → select sentences with higher rank.")

st.header("Text Summarizer Method")
st.write("There are many techniques available to generate extractive summarization. To keep it simple, we will be using an unsupervised learning approach to find the sentences similarity and rank them. One benefit of this will be, you don’t need to train and build a model prior start using it for your project.")
st.write("It’s good to understand Cosine similarity to make the best use of code you are going to see. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. Since we will be representing our sentences as the bunch of vectors, we can use it to find the similarity among sentences. Its measures cosine of the angle between vectors. Angle will be 0 if sentences are similar.")
st.subheader("Simulation")

txt = st.text_area('','Enter Text to be Summarized Here')
number = st.number_input('Insert the number of sentences required in the summary', 1)
def read_article(file_name):
    article = file_name.split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)
    # print("1================")
    # print(sentences)
    df = pd.DataFrame(sentences)
    st.subheader("Generate clean sentences")
    st.write(df)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    st.subheader("Generate Similary Martix across sentences")
    st.write("This is where we will be using cosine similarity to find similarity between sentences.")
    st.dataframe(data= sentence_similarity_martix)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    #print(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # print("===========================")
    st.subheader(" Rank sentences in similarity matrix")
    df1 = pd.DataFrame([scores][0], index = [0])
    st.dataframe(df1)
    # print(scores)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True) 
    #print(' '.join(ranked_sentence[1][1]))
    list1 = []
    for each_sen in ranked_sentence :
        list2 = []
        list2.append(each_sen[0])
        joined = ' '.join(each_sen[1])   
        list2.append(joined)
        list1.append(list2)
    df2 = pd.DataFrame(list1, columns = ['Similarity Score', 'Sentence'])
    st.write("Indexes of top ranked_sentence order are ", df2.head(top_n))    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    st.header("Summarized Text")
    st.write("", ". ".join(summarize_text))

# let's begin
generate_summary( txt, number)