# Combat-online-plagiarism-with-ai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return ""

def detect_plagiarism(original_text, suspect_text):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([original_text, suspect_text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return similarity * 100  # return percentage

def main():
    print("🕵️ AI Plagiarism Detection Tool\n")

    original = read_file("samples/original.txt")
    suspect = read_file("samples/suspect.txt")

    if not original or not suspect:
        print("❗ Please check that both files exist and contain text.")
        return

    similarity_percentage = detect_plagiarism(original, suspect)

    print(f"📊 Similarity: {similarity_percentage:.2f}%")
    if similarity_percentage > 80:
        print("⚠️ High chance of plagiarism.")
    elif similarity_percentage > 50:
        print("🔍 Some content may be similar.")
    else:
        print("✅ Content appears original.")

if __name__ == "__main__":
    main()
