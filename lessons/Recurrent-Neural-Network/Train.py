# Runner for the "AI writer" examples: train a character-level RNN on the
# corpora committed in this folder and watch it generate text every epoch.
#
# Corpora:
#   Text.txt     - an English article about machine learning
#   Template.html / Index.html - small HTML files (teach the model to write HTML)
#
# Generated text is written to the temp/ folder after every epoch.

from Writer import build_model1, build_model2, train_model


def load_content(file_name):
    with open(file_name, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    print("+++++ Example 1: writing an article (LSTM) +++++")
    content = load_content("Text.txt")
    train_model(content, max_seqlen=20, build_model=build_model1,
                write_tofilename="article.txt", num_epochs=30)

    print("+++++ Example 2: writing HTML code (GRU) +++++")
    content = load_content("Template.html")
    train_model(content, max_seqlen=20, build_model=build_model2,
                write_tofilename="html.txt", num_epochs=30)
