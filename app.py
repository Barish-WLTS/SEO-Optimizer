from flask import Flask, render_template, request
import re
import google.generativeai as genai

app = Flask(__name__)

def parse_keywords_links(keywords_text):
    """
    Parses the input keywords text into a list of (keyword, URL) tuples.
    """
    articles = []
    lines = keywords_text.splitlines()
    current_article_keywords = []
    url_regex = re.compile(r'https?://\S+')
    line_split_regex = re.compile(r'\s{2,}|\t+')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if url_regex.search(line) or line.endswith('[No Hyperlink]'):
            parts = line_split_regex.split(line)
            if len(parts) >= 2:
                keyword = parts[0].strip()
                url_part = parts[1].strip()
                url = None if url_part.lower() == '[no hyperlink]' else url_part
                current_article_keywords.append((keyword, url))
    
    if current_article_keywords:
        articles.append(current_article_keywords)

    return articles


def generate_prompt(word_count, dialect, topic, keywords_links, add_context, context, remove_keywords, bulletpoints):
    """
    Generates a prompt string based on provided parameters.
    """
    keywords_formatted = ""
    for keyword, url in keywords_links:
        if url:
            keywords_formatted += f"- **{keyword}**: [{url}]\n"
        else:
            keywords_formatted += f"- **{keyword}**: [No Hyperlink]\n"

    context_part = f"Analyze and learn this content {context.strip()}\n" if add_context and context.strip() else ""

    instruction = "Give it an appropriate title."
    if bulletpoints:
        instruction += " Add bulletpoints if necessary."

    keywords_section = f"Naturally integrate the following keywords only once, each hyperlinked to their respective URLs:\n\n{keywords_formatted}\n" if not remove_keywords else ""

    base_prompt = f"""{context_part}Research and compose a {word_count}-word article in SEO format with subheadings on the topic of "{topic}". Write in {dialect} English. {instruction}
{keywords_section}

Structure the article with a compelling introduction, a well-developed body, and a concise conclusion. Use varied sentence structures and a diverse vocabulary to maintain reader interest and convey information effectively."""

    return base_prompt.strip()


@app.route('/', methods=['GET', 'POST'])
def index():
    article_text = None
    error_message = None

    if request.method == 'POST':
        gemini_api_key = request.form.get("gemini_api_key")
        word_count = int(request.form.get("word_count"))
        dialect = request.form.get("dialect")
        topic = request.form.get("topic")
        keywords_text = request.form.get("keywords_text")
        add_context_option = request.form.get("add_context") == "Yes"
        context_input = request.form.get("context") if add_context_option else ""
        remove_keywords = request.form.get("remove_keywords") == "Yes"
        bulletpoints = request.form.get("bulletpoints") == "Yes"

        if not gemini_api_key:
            error_message = "Please enter your Gemini API Key."
        elif not topic.strip():
            error_message = "Please enter a topic."
        elif not keywords_text.strip():
            error_message = "Please enter keywords and URLs."
        else:
            parsed_keywords = parse_keywords_links(keywords_text)
            if not parsed_keywords:
                error_message = "No valid keywords found."
            else:
                keywords_links = parsed_keywords[0]
                prompt = generate_prompt(
                    word_count, dialect, topic, keywords_links,
                    add_context_option, context_input, remove_keywords, bulletpoints
                )

                try:
                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
                    response = model.generate_content(prompt)
                    article_text = response.text
                except Exception as e:
                    error_message = f"Error generating article with Gemini AI: {e}"

    return render_template("index.html", article_text=article_text, error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
