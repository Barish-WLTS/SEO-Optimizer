from flask import Flask, render_template, request, redirect, url_for, send_file, session,Response
import re
import google.generativeai as genai
from dotenv import load_dotenv
import os
from io import BytesIO
import base64
import hashlib
import datetime
from docx import Document
import pandas as pd
from PIL import Image, UnidentifiedImageError, ImageDraw
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from pathlib import Path
import piexif
import markdown



app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or 'your-secret-key-here'

# Configure upload folder using pathlib
UPLOAD_FOLDER = Path('uploads').absolute()
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

def generate_file_hash(file_data):
    """Generate MD5 hash for file content."""
    return hashlib.md5(file_data).hexdigest()


def is_valid_image(file_data):
    """Check if file is a valid image."""
    try:
        with BytesIO(file_data) as img_bytes:
            img = Image.open(img_bytes)
            img.verify()
        return True
    except (UnidentifiedImageError, IOError):
        return False


def can_piexif_edit(file_ext):
    """Check if file type can be edited by piexif."""
    editable_exts = {'.jpg', '.jpeg'}
    return file_ext.lower() in editable_exts


def read_exif_tags_piexif(file_bytes):
    """Read EXIF data using piexif."""
    try:
        with BytesIO(file_bytes) as img_bytes:
            img = Image.open(img_bytes)
            if 'exif' in img.info:
                exif_dict = piexif.load(img.info['exif'])
                return {
                    'Title': exif_dict.get('0th', {}).get(piexif.ImageIFD.ImageDescription, b'').decode('utf-8', errors='ignore'),
                    'Author': exif_dict.get('0th', {}).get(piexif.ImageIFD.Artist, b'').decode('utf-8', errors='ignore'),
                    'Copyright': exif_dict.get('0th', {}).get(piexif.ImageIFD.Copyright, b'').decode('utf-8', errors='ignore'),
                    'DateTimeOriginal': exif_dict.get('Exif', {}).get(piexif.ExifIFD.DateTimeOriginal, b'').decode('utf-8', errors='ignore'),
                    'GPSLatitude': exif_dict.get('GPS', {}).get(piexif.GPSIFD.GPSLatitude),
                    'GPSLongitude': exif_dict.get('GPS', {}).get(piexif.GPSIFD.GPSLongitude)
                }
        return {}
    except Exception as e:
        print(f"Error reading EXIF with piexif: {str(e)}")
        return {}


def write_exif_tags_piexif(file_bytes, orig_filename, **kwargs):
    """Write EXIF data using piexif."""
    try:
        file_ext = Path(orig_filename).suffix
        if not can_piexif_edit(file_ext):
            final_name = kwargs.get(
                'new_filename', '').strip() or orig_filename
            return file_bytes, final_name

        with BytesIO(file_bytes) as img_bytes:
            img = Image.open(img_bytes)

            # Load existing EXIF or create new
            exif_dict = {}
            if 'exif' in img.info:
                exif_dict = piexif.load(img.info['exif'])
            else:
                exif_dict['0th'] = {}
                exif_dict['Exif'] = {}
                exif_dict['GPS'] = {}
                exif_dict['1st'] = {}
                exif_dict['thumbnail'] = None

            # Update EXIF data based on form inputs
            if kwargs.get('description', '').strip():
                exif_dict['0th'][piexif.ImageIFD.ImageDescription] = kwargs['description'].encode(
                    'utf-8')

            if kwargs.get('keywords'):
                keywords = kwargs['keywords']
                if isinstance(keywords, str):
                    keyword_list = [kw.strip() for kw in keywords.split(',')]  # Split by comma and remove extra spaces
                    keywords = ';'.join(keyword_list)  # Join with semicolon as separator

                exif_dict['0th'][piexif.ImageIFD.XPKeywords] = keywords.encode('utf-16le')  # Store as UTF-16 for compatibility


            # if kwargs.get('author', '').strip():
            #     exif_dict['0th'][piexif.ImageIFD.Artist] = kwargs['author'].encode('utf-8')

            # if kwargs.get('copyright', '').strip():
            #     exif_dict['0th'][piexif.ImageIFD.Copyright] = kwargs['copyright'].encode('utf-8')

            # if kwargs.get('date_taken', '').strip():
            #     exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = kwargs['date_taken'].encode('utf-8')

            # Handle GPS coordinates if provided
            if kwargs.get('latitude') is not None and kwargs.get('longitude') is not None:
                latitude = float(kwargs['latitude'])
                longitude = float(kwargs['longitude'])
                
                # Convert to EXIF GPS format (degrees, minutes, seconds as tuples)
                def decimal_to_dms(decimal):
                    degrees = int(decimal)
                    remainder = abs(decimal - degrees) * 60
                    minutes = int(remainder)
                    seconds = (remainder - minutes) * 60
                    
                    # Convert seconds to a fraction with a large denominator (1,000,000 for microsecond precision)
                    seconds_numerator = int(round(seconds * 1_000_000))  # Avoid floating-point inaccuracies
                    return [
                        (degrees, 1),            # Degrees (exact)
                        (minutes, 1),            # Minutes (exact)
                        (seconds_numerator, 1_000_000)  # Seconds (microsecond precision)
                    ]

                exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = decimal_to_dms(
                    abs(latitude))
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = decimal_to_dms(
                    abs(longitude))
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = 'N' if latitude >= 0 else 'S'
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = 'E' if longitude >= 0 else 'W'

            # Save the image with updated EXIF data
            exif_bytes = piexif.dump(exif_dict)
            output = BytesIO()
            img.save(output, format='jpeg', exif=exif_bytes)
            updated_file_data = output.getvalue()

            final_name = kwargs.get(
                'new_filename', '').strip() or orig_filename
            return updated_file_data, final_name

    except Exception as e:
        print(f"Error writing EXIF with piexif: {str(e)}")
        return file_bytes, orig_filename


@app.route('/exif-editor', methods=['GET', 'POST'])
def exif_editor():
    current_files = []
    selected_file = None
    selected_file_data = None
    exif_data = {}

    # Initialize file names in session if not exists
    if 'file_names' not in session:
        session['file_names'] = {}

    # Ensure upload folder exists
    UPLOAD_FOLDER.mkdir(exist_ok=True)

    if request.method == 'POST':
        # File Upload Logic
        if 'upload_files' in request.files:
            files = request.files.getlist('upload_files')
            for file in files:
                if file.filename:
                    try:
                        file_data = file.read()

                        if len(file_data) > app.config['MAX_CONTENT_LENGTH']:
                            continue

                        file_hash = generate_file_hash(file_data)
                        safe_filename = secure_filename(file.filename)
                        file_path = UPLOAD_FOLDER / file_hash

                        with open(file_path, 'wb') as f:
                            f.write(file_data)

                        # Store original filename in session
                        session['file_names'][file_hash] = safe_filename
                        session.modified = True
                    except Exception as e:
                        print(f"Error saving file {file.filename}: {str(e)}")
                        continue

        # Clear gallery
        if 'clear_gallery' in request.form:
            for file_path in UPLOAD_FOLDER.glob('*'):
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error deleting file {file_path}: {str(e)}")
            session['file_names'] = {}
            session.modified = True

        # Select file for editing
        if 'select_file' in request.form:
            selected_file_hash = request.form['select_file']
            session['selected_file'] = selected_file_hash
            session.modified = True

        # Get selected file from session
        selected_file_hash = session.get('selected_file')

        if selected_file_hash:
            selected_file_path = UPLOAD_FOLDER / selected_file_hash

            if selected_file_path.exists():
                try:
                    with open(selected_file_path, 'rb') as f:
                        selected_file_data = f.read()

                    selected_file = {
                        'hash': selected_file_hash,
                        'filename': session['file_names'].get(selected_file_hash, selected_file_hash),
                        'size': selected_file_path.stat().st_size
                    }

                    exif_data = read_exif_tags_piexif(selected_file_data)
                except Exception as e:
                    print(f"Error reading file {selected_file_path}: {str(e)}")

        # Write EXIF tags
        if 'write_exif' in request.form and selected_file_hash:
            selected_file_path = UPLOAD_FOLDER / selected_file_hash
            if selected_file_path.exists():
                try:
                    with open(selected_file_path, 'rb') as f:
                        file_data = f.read()

                    new_filename = request.form.get(
                        'new_filename', selected_file['filename'])
                    description = request.form.get('description', '')
                    keywords = request.form.get('keywords', '')
                    # author = request.form.get('author', '')
                    # date_taken = request.form.get('date_taken', '')
                    # copyright_text = request.form.get('copyright', '')

                    try:
                        latitude = float(request.form.get('latitude', 0))
                        longitude = float(request.form.get('longitude', 0))
                    except (TypeError, ValueError):
                        latitude, longitude = None, None

                    updated_file_data, final_filename = write_exif_tags_piexif(
                        file_data,
                        selected_file['filename'],
                        new_filename=new_filename,
                        description=description,
                        keywords=keywords,
                        # author=author,
                        # date_taken=date_taken,
                        # copyright=copyright_text,
                        latitude=latitude,
                        longitude=longitude
                    )

                    with open(selected_file_path, 'wb') as f:
                        f.write(updated_file_data)

                    # Update the filename in session if it was changed
                    if new_filename and new_filename != selected_file['filename']:
                        session['file_names'][selected_file_hash] = final_filename
                        session.modified = True

                    # Re-read the file to get updated EXIF data
                    with open(selected_file_path, 'rb') as f:
                        selected_file_data = f.read()
                    exif_data = read_exif_tags_piexif(selected_file_data)

                    # Update selected_file with new filename
                    selected_file = {
                        'hash': selected_file_hash,
                        'filename': session['file_names'].get(selected_file_hash, selected_file_hash),
                        'size': selected_file_path.stat().st_size
                    }

                except Exception as e:
                    print(
                        f"Error processing file {selected_file_path}: {str(e)}")

    # List all files in upload folder
    for file_path in UPLOAD_FOLDER.glob('*'):
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()

            current_files.append({
                'hash': file_path.name,
                'filename': session['file_names'].get(file_path.name, file_path.name),
                'is_image': is_valid_image(file_data),
                'size': file_path.stat().st_size
            })
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            continue

    return render_template('exif_editor.html',
                           current_files=current_files,
                           selected_file=selected_file,
                           selected_file_data=selected_file_data,
                           exif_data=exif_data,
                           base64=base64)


@app.route('/download/<file_hash>')
def download_file(file_hash):
    file_path = UPLOAD_FOLDER / file_hash
    if file_path.exists():
        filename = session['file_names'].get(file_hash, file_hash)
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    return "File not found", 404


@app.route('/thumbnail/<file_hash>')
def get_thumbnail(file_hash):
    file_path = UPLOAD_FOLDER / file_hash
    if file_path.exists():
        try:
            with open(file_path, 'rb') as f:
                img_data = f.read()

            if is_valid_image(img_data):
                img = Image.open(BytesIO(img_data))
                img.thumbnail((200, 200))

                output = BytesIO()
                img.save(output, format='JPEG')
                return Response(output.getvalue(), mimetype='image/jpeg')
        except Exception as e:
            print(f"Error generating thumbnail: {str(e)}")

    # Return a placeholder if we can't generate a thumbnail
    img = Image.new('RGB', (200, 200), color='gray')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "No preview", fill='white')

    output = BytesIO()
    img.save(output, format='JPEG')
    return Response(output.getvalue(), mimetype='image/jpeg')


# Geotag-ends-here

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
                url = None if url_part.lower(
                ) == '[no hyperlink]' else url_part
                current_article_keywords.append((keyword, url))

    if current_article_keywords:
        articles.append(current_article_keywords)

    return articles


def clean_article_text(article_text):
    """
    Cleans the article text, adds spacing between paragraphs, and preserves HTML.

    Args:
        article_text: The article text string to clean.

    Returns:
        The cleaned article text string with HTML intact and spacing.
    """

    # Remove starting ```html and ending ``` markers (if they exist - be more forgiving)
    text = re.sub(r'(^```html\s*|\s*```$)', '',
                  article_text, flags=re.MULTILINE)

    # Add extra newline after each paragraph tag  (Improved spacing)
    # Two newlines after each paragraph
    text = re.sub(r'</p>', '</p>\n\n', text)

    # Ensure headings have appropriate spacing around them
    # Adds two newlines *before* and *after* headings
    text = re.sub(r'(<h[1-6]>.*?</h[1-6]>)', r'\n\n\1\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()
    return text


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

    context_part = f"Analyze and learn this content {context.strip()}\n" if add_context and context.strip(
    ) else ""

    instruction = "Give it an appropriate title."
    if bulletpoints:
        instruction += " Add bulletpoints if necessary."

    keywords_section = f"Naturally integrate the following keywords only once, each hyperlinked to their respective URLs:\n\n{keywords_formatted}\n" if not remove_keywords else ""

    base_prompt = f"""{context_part}Research and compose a {word_count}-word article in SEO format with subheadings on the topic of "{topic}". Write in {dialect} English.
{keywords_section}

Structure the article with a compelling introduction, a well-developed body, and a concise conclusion. Use varied sentence structures and a diverse vocabulary to maintain reader interest and convey information effectively. {instruction}"""

    return base_prompt.strip()



@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")


def split_into_sections(text, keywords):
    """Splits the article text into sections based on headings (HTML or Markdown).
    If no clear sections, creates a single section.
    """
    sections = []
    text = text.strip()

    if not text:
        return sections

    # Convert Markdown to HTML if needed
    if not re.search(r'<h[1-6]>', text):  # If no HTML headings found
        text = markdown.markdown(text)  # Convert Markdown to HTML

    # Split by HTML headings
    heading_sections = re.split(
        r'(<h[1-6]>.*?</h[1-6]>)', text, flags=re.IGNORECASE)

    if len(heading_sections) > 1:  # Found headings
        for i in range(0, len(heading_sections), 2):
            content = heading_sections[i].strip()
            title = ""
            if i + 1 < len(heading_sections):
                title = re.sub('<[^>]*>', '', heading_sections[i+1]).strip()
            if content or title:
                sections.append({
                    'title': title or "Introduction",
                    'content': content,
                    'html': True  # Flag indicating HTML content
                })
        return sections

    # If no headings found, create a single section
    return [{
        'title': "Article Content",
        'content': text,
        'html': True
    }]


@app.route('/article', methods=['GET', 'POST'])
def index():
    error_message = request.args.get(
        'error_message')  # Get error from query params
    # Get and clear article from session
    article_text = session.pop('article_text', None)
    sections = session.pop('sections', None)
    if request.method == 'POST':
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        word_count = int(request.form.get("word_count"))
        dialect = request.form.get("dialect")
        topic = request.form.get("topic")
        keywords_text = request.form.get("keywords_text")
        add_context_option = request.form.get("add_context") == "Yes"
        context_input = request.form.get(
            "context") if add_context_option else ""
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
                    model = genai.GenerativeModel(
                        'gemini-2.0-flash-thinking-exp-01-21')
                    response = model.generate_content(prompt)
                    article_text = response.text

                    # Clean and split the article
                    cleaned_text = clean_article_text(article_text)
                    sections = split_into_sections(
                        cleaned_text, keywords_links)
                    print(sections)
                    # Store in session and redirect
                    session['sections'] = sections
                    return redirect(url_for('index'))
                except Exception as e:
                    error_message = f"Error generating article with Gemini AI: {e}"
                    return redirect(url_for('index', error_message=error_message))

    return render_template("article.html", sections=sections, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
