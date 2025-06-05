from sentence_transformers import util
from nltk.tokenize import sent_tokenize

from embedding_model import get_embedding_model


def main():
    model = get_embedding_model()
    
    text = """
    Hôm nay tôi ăn cơm.
    cơm tôi ăn rất ngon, nó có canh, cua, tôm cá.
    đúng là một bữa cơm tuyệt vời.
    tôi đi xe tới trường.
    """
    
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)
    
    # chunk by simi thresh
    threshold = 0.5
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i], embeddings[i-1]).item()
        print(sim)
        if sim > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            
    # last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    for idx, chunk in enumerate(chunks):
        print(f"\nChunk {idx+1}:\n{chunk}")

if __name__ == "__main__":
    main()