
# Inspiration Search Engine

This project is an Inspiration Search Engine that allows users to find visually similar images based on a textual query. The application utilizes OpenAI's CLIP model for image and text feature extraction, and ChromaDB for managing image embeddings.

## Features

- **Search for Inspiration:** Users can enter a theme or idea to receive visually similar images.
- **Adjustable Results:** Users can specify the number of results to display and set a similarity threshold.
- **Dynamic Image Fetching:** The application fetches images from specified URLs and processes them in real-time.

## Technologies Used

- **Python**: Programming language for building the application.
- **Transformers**: For using the CLIP model for image and text embeddings.
- **ChromaDB**: For storing and querying image embeddings.
- **Gradio**: For creating the web interface.
- **PIL**: For image processing.
- **Requests**: For fetching images from the web.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/uweeeem/Chromadb.git
   cd Chromadb
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install torch torchvision transformers gradio chromadb requests scikit-learn
   ```

## Usage

To run the application, execute the following command:

```bash
python app.py
```

This will start a web server, and you will see a URL in the console output. Open that URL in your browser to access the Inspiration Search Engine.

### Example Queries

- Modern living room
- Minimalist kitchen design
- Outdoor garden ideas



## Video Demonstration

You can watch a demonstration of the application [here](https://youtu.be/lLEenMOYr5w).

Or click the thumbnail below to watch:

[![Watch the video](https://img.youtube.com/vi/lLEenMOYr5w/0.jpg)](https://youtu.be/lLEenMOYr5w)


## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

