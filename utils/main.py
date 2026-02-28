# main.py
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

# -------------------------------
# N-gram feature extraction
# -------------------------------
def ngram(tokens, n):
    output = []
    for i in range(n-1, len(tokens)):
        ngram_ = ' '.join(tokens[i-n+1:i+1])
        output.append(ngram_)
    return output

def create_feature(text, nrange=(1, 4)):
    text_features = []
    text = text.lower()
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1):
        text_features += ngram(text_alphanum.split(), n)
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

# -------------------------------
# Label conversion
# -------------------------------
def convert_label(item, emotions):
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)):
        if items[idx] == 1:
            label += emotions[idx] + " "
    return label.strip()

# -------------------------------
# Read data
# -------------------------------
def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

# -------------------------------
# Train and evaluate classifiers
# -------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    classifiers = [
        SVC(),
        LinearSVC(random_state=123),
        RandomForestClassifier(random_state=123),
        DecisionTreeClassifier()
    ]

    print("| {:25} | {:17} | {:13} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
    print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))

    trained_models = {}
    for clf in classifiers:
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        print("| {:25} | {:17.7f} | {:13.7f} |".format(clf.__class__.__name__, train_acc, test_acc))
        trained_models[clf.__class__.__name__] = clf

    return trained_models

# -------------------------------
# Sample Predictions
# -------------------------------
def predict_samples(model, vectorizer, sample_texts, emoji_dict):
    for text in sample_texts:
        features = create_feature(text, nrange=(1,4))
        features_vector = vectorizer.transform([features])
        prediction = model.predict(features_vector)[0]
        print(text, emoji_dict.get(prediction, "❓"))

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    # File path to your labeled text data
    file_path = r'C:\Users\ssp1_\Downloads\textemotion.txt'

    emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢",
                  "disgust":"😒", "shame":"😳", "guilt":"😳"}

    # Read and process data
    data = read_data(file_path)
    print(f"Number of instances: {len(data)}")

    X_all, y_all = [], []
    for label, text in data:
        y_all.append(convert_label(label, emotions))
        X_all.append(create_feature(text, nrange=(1,4)))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=123)

    # Vectorize features
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train classifiers
    trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Predict sample sentences using Random Forest (you can change)
    sample_texts = [
        "This looks so impressive",
        "I have a fear of dogs",
        "My dog died yesterday",
        "I don't love you anymore..!"
    ]
    print("\nSample Predictions:")
    predict_samples(trained_models["RandomForestClassifier"], vectorizer, sample_texts, emoji_dict)
